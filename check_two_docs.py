"""
Check contextual similarity between two documents.

═══════════════════════════════════════════════════════════════════════
WORKING THEORY / ALGORITHM
═══════════════════════════════════════════════════════════════════════

Step 1 — Text Embedding (Vectorization)
  Raw description text is converted into a dense 384-dimensional vector
  using a pre-trained sentence transformer model (all-MiniLM-L6-v2).
  This model was chosen for its speed/quality tradeoff:
    - ~750 sentences/sec on CPU
    - STS benchmark score: 79.86 (good for semantic similarity)
    - 384-dim output (compact, fast to compare)

  The model maps semantically similar sentences to nearby points in
  384-dim space. E.g. "telecom tower deployment" and "network
  infrastructure setup" end up close together.

Step 2 — L2 Normalization
  Each embedding vector is normalized to unit length (L2 norm = 1).
  This makes the dot product between two vectors equal to their cosine
  similarity — no need to divide by magnitudes separately.

Step 3 — Cosine Similarity (Dot Product)
  Similarity = dot(vec_a, vec_b)

  Since vectors are already L2-normalized:
    - Identical meaning → 1.0
    - Orthogonal (unrelated) → 0.0
    - Opposite meaning → -1.0 (rare in practice)

  The result is clamped to [0.0, 1.0] for interpretability.

Step 4 — Threshold Comparison (optional)
  If a threshold is provided:
    similarity >= threshold → match = True  (contextually similar)
    similarity <  threshold → match = False (not similar)

  Recommended thresholds for this dataset:
    - 0.75+  : Near-duplicate / same event
    - 0.50-0.75: Related topic / same domain
    - 0.30-0.50: Loosely related
    - <0.30  : Unrelated

Two modes:
  1. By ES doc IDs — fetches pre-computed desc_embedding from ES, computes cosine similarity
  2. By raw description text — embeds on-the-fly with sentence-transformers, computes cosine similarity

Usage (CLI):
    python check_two_docs.py --id1 <id1> --id2 <id2> [--threshold 0.5]
    python check_two_docs.py --text1 "description text 1" --text2 "description text 2" [--threshold 0.7]
    python check_two_docs.py <id1> <id2> [threshold]

If threshold is provided, prints YES/NO and exits with code 0/1.
If threshold is omitted, just prints the similarity score.

Usage (import):
    from check_two_docs import check_by_id, check_by_text

    result = check_by_id("abc123", "xyz789")
    print(result["similarity"])  # 0.8234

    result = check_by_text("Network infrastructure setup", "Telecom tower deployment", threshold=0.5)
    # {"similarity": 0.8234, "text_1": "...", "text_2": "...", "threshold": 0.5, "match": True}
"""

import sys
import argparse
import warnings
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

from elasticsearch import Elasticsearch
import numpy as np

# ── Config ──────────────────────────────────────────────────────────
ES_HOST = "192.168.1.125"
ES_PORT = 9200
ES_SCHEME = "https"
ES_USER = "elastic"
ES_PASS = "30oIsFcjJa8Zao+iq5*e"
ES_INDEX = "vec_fatboy_data"
EMBEDDING_FIELD = "desc_embedding"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_es():
    return Elasticsearch(
        [{"host": ES_HOST, "port": ES_PORT, "scheme": ES_SCHEME}],
        basic_auth=(ES_USER, ES_PASS),
        verify_certs=False,
        ssl_show_warn=False,
    )


# ── Function 1: Check by ES Doc IDs ─────────────────────────────────

def check_by_id(doc_id_1: str, doc_id_2: str, threshold: float = None) -> dict:
    """
    Check contextual similarity between two ES docs by their IDs.

    Uses pre-computed desc_embedding stored in ES (from reembed_docs.py).
    Fast — just fetches vectors and computes dot product.

    Args:
        doc_id_1: First document ID
        doc_id_2: Second document ID
        threshold: Optional similarity threshold (0-1). If provided, adds "match" bool.

    Returns:
        {
            "id_1": "...",
            "id_2": "...",
            "similarity": 0.8234,
            "description_1": "...",
            "description_2": "...",
            "threshold": 0.5,   # only if threshold was passed
            "match": True,      # only if threshold was passed
        }
    """
    es = get_es()

    # Fetch both docs
    try:
        doc1 = es.get(index=ES_INDEX, id=doc_id_1)
    except Exception:
        raise ValueError(f"Doc '{doc_id_1}' not found in index '{ES_INDEX}'")

    try:
        doc2 = es.get(index=ES_INDEX, id=doc_id_2)
    except Exception:
        raise ValueError(f"Doc '{doc_id_2}' not found in index '{ES_INDEX}'")

    src1 = doc1["_source"]
    src2 = doc2["_source"]

    emb1 = src1.get(EMBEDDING_FIELD)
    emb2 = src2.get(EMBEDDING_FIELD)

    if emb1 is None:
        raise ValueError(f"Doc '{doc_id_1}' has no '{EMBEDDING_FIELD}' field. Run reembed_docs.py first.")
    if emb2 is None:
        raise ValueError(f"Doc '{doc_id_2}' has no '{EMBEDDING_FIELD}' field. Run reembed_docs.py first.")

    # Cosine similarity (vectors already normalized)
    vec1 = np.array(emb1, dtype=np.float32)
    vec2 = np.array(emb2, dtype=np.float32)
    similarity = round(float(np.dot(vec1, vec2)), 4)

    result = {
        "id_1": doc_id_1,
        "id_2": doc_id_2,
        "similarity": similarity,
        "description_1": (src1.get("description") or "").strip(),
        "description_2": (src2.get("description") or "").strip(),
    }

    if threshold is not None:
        result["threshold"] = threshold
        result["match"] = similarity >= threshold

    return result


# ── Function 2: Check by Raw Description Text ───────────────────────

def check_by_text(text_1: str, text_2: str, threshold: float = None, model_name: str = DEFAULT_MODEL) -> dict:
    """
    Check contextual similarity between two raw description strings.

    Embeds on-the-fly using sentence-transformers (all-MiniLM-L6-v2).
    Slower than check_by_id but works without ES embeddings.

    Args:
        text_1: First description text
        text_2: Second description text
        threshold: Optional similarity threshold (0-1). If provided, adds "match" bool.
        model_name: Sentence transformer model to use.

    Returns:
        {
            "text_1": "...",
            "text_2": "...",
            "similarity": 0.8234,
            "threshold": 0.5,   # only if threshold was passed
            "match": True,      # only if threshold was passed
        }
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        [text_1, text_2],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    similarity = round(float(np.dot(embeddings[0], embeddings[1])), 4)

    result = {
        "text_1": text_1,
        "text_2": text_2,
        "similarity": similarity,
    }

    if threshold is not None:
        result["threshold"] = threshold
        result["match"] = similarity >= threshold

    return result


# ── CLI ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Check contextual similarity between two documents"
    )
    parser.add_argument("id1", nargs="?", help="First doc ID (positional)")
    parser.add_argument("id2", nargs="?", help="Second doc ID (positional)")
    parser.add_argument("--id1", dest="id1_flag", help="First doc ID (flag form)")
    parser.add_argument("--id2", dest="id2_flag", help="Second doc ID (flag form)")
    parser.add_argument("--text1", help="Raw description text 1 (instead of doc IDs)")
    parser.add_argument("--text2", help="Raw description text 2 (instead of doc IDs)")
    parser.add_argument("--threshold", "-t", type=float, default=None,
                        help="Similarity threshold (0-1). Prints YES/NO, exits 0/1")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL,
                        help="Sentence transformer model (for --text mode)")
    args = parser.parse_args()

    # Determine mode
    if args.text1 and args.text2:
        # Text mode
        threshold = args.threshold
        result = check_by_text(args.text1, args.text2, threshold=threshold, model_name=args.model)
        print(f"Text 1: {result['text_1'][:100]}")
        print(f"Text 2: {result['text_2'][:100]}")
        print(f"\nSimilarity: {result['similarity']:.4f}")
        if threshold is not None:
            if result["match"]:
                print(f"RESULT: YES (similar, >= {threshold})")
                sys.exit(0)
            else:
                print(f"RESULT: NO (not similar, < {threshold})")
                sys.exit(1)

    elif (args.id1 or args.id1_flag) and (args.id2 or args.id2_flag):
        # ID mode
        id1 = args.id1 or args.id1_flag
        id2 = args.id2 or args.id2_flag
        threshold = args.threshold

        result = check_by_id(id1, id2, threshold=threshold)
        print(f"Doc 1: {result['id_1']}")
        print(f"  desc: {result['description_1'][:100]}")
        print(f"Doc 2: {result['id_2']}")
        print(f"  desc: {result['description_2'][:100]}")
        print(f"\nSimilarity: {result['similarity']:.4f}")
        if threshold is not None:
            if result["match"]:
                print(f"RESULT: YES (similar, >= {threshold})")
                sys.exit(0)
            else:
                print(f"RESULT: NO (not similar, < {threshold})")
                sys.exit(1)

    else:
        print("Usage:")
        print("  By ES IDs:   python check_two_docs.py <id1> <id2> [threshold]")
        print("  By ES IDs:   python check_two_docs.py --id1 <id1> --id2 <id2> [--threshold 0.5]")
        print("  By text:     python check_two_docs.py --text1 \"desc1\" --text2 \"desc2\" [--threshold 0.7]")
        sys.exit(2)


if __name__ == "__main__":
    main()
