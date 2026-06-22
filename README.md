# Weitage Similarity — Contextual Document Similarity Engine

Automatically assign weitage scores (0.1–0.9) to Elasticsearch documents based on contextual similarity of their `description` field, using sentence-transformer embeddings.

---

## How It Works

### Algorithm

1. **Text Embedding** — Each document's `description` is converted to a 384-dimensional vector using `all-MiniLM-L6-v2` (sentence-transformers). This model maps semantically similar sentences to nearby points in vector space.
2. **L2 Normalization** — Vectors are normalized to unit length, so the dot product directly equals cosine similarity.
3. **Cosine Similarity** — `similarity = dot(vec_a, vec_b)`. Score of 1.0 = identical meaning, 0.0 = unrelated.
4. **Threshold Matching** — If a threshold is provided, docs with similarity >= threshold are considered matches.

### Recommended Thresholds

| Score | Meaning |
|-------|---------|
| 0.75+ | Near-duplicate / same event |
| 0.50–0.75 | Related topic / same domain |
| 0.30–0.50 | Loosely related |
| < 0.30 | Unrelated |

### Two Modes

- **By ES Doc ID** — Fetches pre-computed embeddings from ES (`desc_embedding` field). Fast, no model loading at query time.
- **By Raw Text** — Embeds on-the-fly. Slower but works without ES embeddings.

---

## Files

| File | Purpose |
|------|---------|
| `check_two_docs.py` | Check similarity between two docs (by ID or text). CLI + importable functions. |
| `reembed_docs.py` | Batch-embed all ES docs with sentence-transformers and store vectors in ES. |
| `requirements.txt` | Python dependencies. |

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Usage

### CLI

```bash
# Check similarity by ES doc IDs
python check_two_docs.py --id1 <id1> --id2 <id2> --threshold 0.5

# Check similarity by raw text
python check_two_docs.py --text1 "Network infrastructure" --text2 "Telecom tower setup" --threshold 0.7

# Positional IDs (shorthand)
python check_two_docs.py <id1> <id2> 0.5
```

With `--threshold`: prints YES/NO and exits with code 0 (match) or 1 (no match).
Without `--threshold`: just prints the similarity score.

### Python Import

```python
from check_two_docs import check_by_id, check_by_text

# By ES doc ID (fast, uses pre-computed embeddings)
result = check_by_id("doc_id_1", "doc_id_2", threshold=0.5)
print(result["similarity"])  # 0.8234
print(result["match"])       # True

# By raw text (embeds on-the-fly)
result = check_by_text("Network setup", "Telecom deployment", threshold=0.5)
print(result["similarity"])  # 0.8117
```

---

## Re-embedding ES Docs

Before using `check_by_id`, you need to populate the `desc_embedding` field in your ES index:

```bash
python reembed_docs.py
```

Options:
```bash
python reembed_docs.py --max-docs 100      # limit for testing
python reembed_docs.py --batch-size 200    # bulk update batch size
python reembed_docs.py --dry-run           # don't write to ES
```

This scrolls through all docs in `vec_fatboy_data` with a non-empty `description`, embeds them, and bulk-updates ES with the 384-dim vectors.

---

## Configuration

ES connection settings are at the top of both scripts:

```
Host:     192.168.1.125
Port:     9200
Index:    vec_fatboy_data
Field:    desc_embedding
Model:    all-MiniLM-L6-v2
```

---

## Dependencies

- `elasticsearch>=8.0.0` — ES Python client
- `sentence-transformers>=2.2.0` — all-MiniLM-L6-v2 model
- `numpy>=1.21.0` — vector operations
- `torch>=1.10.0` — required by sentence-transformers
