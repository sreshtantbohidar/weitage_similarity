"""
Script 2: Re-embed all docs in ES with sentence-transformer (all-MiniLM-L6-v2).

Scrolls through all docs in vec_fatboy_data that have a non-empty description,
embeds the description text, and bulk-updates ES with the embedding vector.

Run once (takes ~10-15 min for 35K docs on CPU):
    python reembed_docs.py

Options:
    --batch-size   Number of docs per batch (default 100)
    --max-docs     Max docs to process (default 0 = all)
    --dry-run      Print what would be done without writing to ES
"""

import sys
import time
import argparse
import logging
import warnings
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer
import numpy as np

# ── Config ──────────────────────────────────────────────────────────
ES_HOST = "192.168.1.125"
ES_PORT = 9200
ES_SCHEME = "https"
ES_USER = "elastic"
ES_PASS = "30oIsFcjJa8Zao+iq5*e"
ES_INDEX = "vec_fatboy_data"

EMBEDDING_FIELD = "desc_embedding"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SCROLL_SIZE = 500  # ES scroll batch size
BULK_SIZE = 100    # bulk update batch size


def get_es():
    return Elasticsearch(
        [{"host": ES_HOST, "port": ES_PORT, "scheme": ES_SCHEME}],
        basic_auth=(ES_USER, ES_PASS),
        verify_certs=False,
        ssl_show_warn=False,
    )


def scroll_docs(es, max_docs=0):
    """Scroll through all docs with non-empty description."""
    query = {
        "_source": ["description"],
        "query": {
            "bool": {
                "must": [
                    {"exists": {"field": "description"}},
                    {"range": {"description": {"gte": "a"}}},  # non-empty
                ]
            }
        },
    }

    if max_docs > 0:
        query["size"] = min(max_docs, SCROLL_SIZE)

    result = es.search(index=ES_INDEX, body=query, scroll="5m")
    scroll_id = result.get("_scroll_id")
    hits = result["hits"]["hits"]
    total = result["hits"]["total"]["value"]
    yielded = 0

    print(f"Total docs with description: {total}")
    if max_docs > 0:
        print(f"Limiting to {max_docs} docs")

    while hits:
        for h in hits:
            desc = (h["_source"].get("description") or "").strip()
            if len(desc) > 3:  # skip very short descriptions
                yield h["_id"], desc
                yielded += 1
                if max_docs > 0 and yielded >= max_docs:
                    return

        if scroll_id:
            result = es.scroll(scroll_id=scroll_id, scroll="5m")
            scroll_id = result.get("_scroll_id")
            hits = result["hits"]["hits"]
        else:
            break

    if scroll_id:
        try:
            es.clear_scroll(scroll_id=scroll_id)
        except Exception:
            pass


def flush_bulk(es, actions, dry_run=False):
    """Flush bulk actions to ES."""
    if not actions:
        return 0

    if dry_run:
        print(f"  [DRY RUN] Would update {len(actions)} docs")
        actions.clear()
        return 0

    success, errors = helpers.bulk(
        es,
        actions,
        index=ES_INDEX,
        raise_on_error=False,
        chunk_size=BULK_SIZE,
    )

    err_count = 0
    if isinstance(errors, list):
        err_count = len(errors)
    elif isinstance(errors, int):
        err_count = errors

    if err_count > 0:
        print(f"  WARNING: {err_count} errors during bulk update")

    actions.clear()
    return success


def main():
    parser = argparse.ArgumentParser(description="Re-embed ES docs with sentence-transformers")
    parser.add_argument("--batch-size", type=int, default=BULK_SIZE, help="Bulk update batch size")
    parser.add_argument("--max-docs", type=int, default=0, help="Max docs to process (0=all)")
    parser.add_argument("--dry-run", action="store_true", help="Don't write to ES")
    args = parser.parse_args()

    es = get_es()

    # Check connection
    try:
        info = es.info()
        print(f"Connected to ES: {info['cluster_name']} (v{info['version']['number']})")
    except Exception as e:
        print(f"ES connection failed: {e}")
        sys.exit(1)

    if not es.indices.exists(index=ES_INDEX):
        print(f"Index '{ES_INDEX}' does not exist!")
        sys.exit(1)

    # Load model
    print(f"Loading model: {MODEL_NAME}")
    t0 = time.time()
    model = SentenceTransformer(MODEL_NAME)
    print(f"Model loaded in {time.time() - t0:.1f}s. Dim: {model.get_embedding_dimension()}")

    # Collect docs
    print("\nFetching docs from ES...")
    docs = list(scroll_docs(es, max_docs=args.max_docs))
    print(f"Collected {len(docs)} docs to embed\n")

    if not docs:
        print("No docs to process. Exiting.")
        sys.exit(0)

    if args.dry_run:
        print("=== DRY RUN MODE ===")
        for doc_id, desc in docs[:5]:
            emb = model.encode([desc], normalize_embeddings=True)[0]
            print(f"  {doc_id}: dim={len(emb)}, desc={desc[:60]}")
        if len(docs) > 5:
            print(f"  ... and {len(docs) - 5} more")
        print(f"\nWould process {len(docs)} docs total")
        sys.exit(0)

    # Process in batches
    total_updated = 0
    batch_actions = []
    embed_batch = []
    batch_start = time.time()

    for i, (doc_id, desc) in enumerate(docs):
        embed_batch.append((doc_id, desc))

        if len(embed_batch) >= args.batch_size or i == len(docs) - 1:
            # Embed batch
            descs = [d[1] for d in embed_batch]
            embeddings = model.encode(
                descs,
                show_progress_bar=False,
                normalize_embeddings=True,
                batch_size=64,
            )

            # Build bulk actions
            for (did, _), emb in zip(embed_batch, embeddings):
                batch_actions.append({
                    "_op_type": "update",
                    "_id": did,
                    "doc": {EMBEDDING_FIELD: emb.tolist()},
                })

            # Flush
            flushed = flush_bulk(es, batch_actions)
            total_updated += flushed
            embed_batch.clear()

            # Progress
            elapsed = time.time() - batch_start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            pct = (i + 1) / len(docs) * 100
            remaining = (len(docs) - i - 1) / rate if rate > 0 else 0
            print(
                f"  {i+1}/{len(docs)} ({pct:.1f}%) | "
                f"{total_updated} updated | "
                f"{rate:.0f} docs/s | "
                f"ETA: {remaining:.0f}s"
            )

    elapsed = time.time() - batch_start
    print(f"\n=== DONE ===")
    print(f"Updated {total_updated} docs in {elapsed:.1f}s ({total_updated/elapsed:.1f} docs/s)")

    # Verify: count docs with embedding
    count = es.count(
        index=ES_INDEX,
        body={"query": {"exists": {"field": EMBEDDING_FIELD}}}
    )["count"]
    print(f"Docs with '{EMBEDDING_FIELD}': {count}")


if __name__ == "__main__":
    main()
