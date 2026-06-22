# Weitage Similarity — Contextual Document Similarity Engine

Automatically assign weitage scores (0.1–0.9) to Elasticsearch documents based on contextual similarity of their `description` field, using sentence-transformer embeddings.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Solution Overview](#solution-overview)
3. [Theory & Algorithm (Detailed)](#theory--algorithm-detailed)
4. [End-to-End Flow](#end-to-end-flow)
5. [Two Modes](#two-modes)
6. [Threshold Guide](#threshold-guide)
7. [Files](#files)
8. [Setup](#setup)
9. [Usage](#usage)
10. [Configuration](#configuration)
11. [Dependencies](#dependencies)

---

## Problem Statement

The Elasticsearch index `vec_fatboy_data` contains ~35,000 documents. Some documents have a manually-assigned **weitage** score (0.1–0.9) indicating relevance/priority. The remaining documents need weitage auto-assigned based on their contextual similarity to documents that already have weitage set.

**Key requirement:** Compare documents by their `description` field (free-text), not by exact keyword matching. Two descriptions like _"Telecom tower network setup"_ and _"Network infrastructure deployment"_ should be recognized as similar even though they share no exact words.

---

## Solution Overview

The system uses **Sentence Transformers** (a type of neural network) to convert text into numerical vectors (embeddings) that capture semantic meaning. Similar texts produce similar vectors. We then compare vectors using **cosine similarity** to quantify how contextually related two documents are.

**Model chosen:** `all-MiniLM-L6-v2`
- 384-dimensional embeddings
- ~750 sentences/second on CPU
- STS benchmark score: 79.86 (good speed/quality tradeoff)
- ~90MB download
- Alternative: `all-mpnet-base-v2` for higher accuracy (768-dim, ~170/sec, STS 84.33)

---

## Theory & Algorithm (Detailed)

### Step 1 — Text Embedding (Vectorization)

**What is an embedding?**
An embedding is a list of numbers (a vector) that represents text in a high-dimensional space. Think of it as a "fingerprint" of the text's meaning.

**How it works:**
The `all-MiniLM-L6-v2` model is a neural network (based on BERT/distilBERT) trained on millions of sentence pairs to understand semantic relationships. When you feed it a sentence:

1. The text is tokenized into sub-words (e.g., "deployment" → "deploy" + "ment")
2. Each token is converted to an embedding via lookup tables
3. Transformer layers process all tokens together, capturing context and relationships
4. A mean-pooling layer averages all token embeddings into a single 384-dim vector

**Why this captures "contextual similarity":**
The model was trained so that sentences with similar meanings end up close together in 384-dim space. For example:
- "Telecom tower network setup" → [0.023, -0.118, 0.042, ...] (384 numbers)
- "Network infrastructure deployment" → [0.031, -0.095, 0.038, ...] (384 numbers)
- "Employee payroll management" → [-0.214, 0.067, -0.103, ...] (384 numbers)

The first two vectors are close (similar meaning). The third is far (different meaning).

**Why all-MiniLM-L6-v2 specifically:**
- **384 dimensions** is compact — fast to store and compare (vs 768 or 1536 for larger models)
- **L6** = 6 transformer layers (vs 12 for BERT-base) — faster but still effective
- **Trained on** 1B+ sentence pairs including NLI, STS, and QA datasets
- **Specifically designed** for semantic similarity tasks (not classification or generation)

### Step 2 — L2 Normalization

**What is it?**
After embedding, each vector is scaled so its total length (L2 norm) equals 1.0.

```
normalized = vector / sqrt(sum(vector[i]^2 for all i))
```

**Why do this?**
- Raw embeddings have varying magnitudes (some vectors are longer than others)
- L2 normalization removes magnitude as a factor — only direction matters
- After normalization: `dot_product(A, B) = cosine_similarity(A, B)`
- This means we can use a simple dot product instead of the full cosine formula

**Example:**
```
Raw vector A: [3.0, 4.0]       → norm = sqrt(9+16) = 5.0
Normalized A: [0.6, 0.8]       → norm = sqrt(0.36+0.64) = 1.0

Raw vector B: [6.0, 8.0]       → norm = 10.0
Normalized B: [0.6, 0.8]       → norm = 1.0

dot(A_norm, B_norm) = 0.6*0.6 + 0.8*0.8 = 1.0  (identical direction = similarity 1.0)
```

### Step 3 — Cosine Similarity (Dot Product)

**The formula:**
```
similarity = (A · B) / (||A|| * ||B||)
```

Since we already L2-normalized (||A|| = 1, ||B|| = 1), this simplifies to:
```
similarity = A · B = sum(A[i] * B[i] for all i in 0..383)
```

**What the scores mean:**

| Score | Interpretation | Example |
|-------|---------------|---------|
| 1.0000 | Identical meaning | Same description text |
| 0.80–0.99 | Near-duplicate / same event | "RR, CRPF exchanged fire with HM militants on Nov 11" vs "On 11 November, RR, CRPF and SoG exchanged fire with HM militants" |
| 0.60–0.80 | Strongly related | "Telecom tower deployment" vs "Network infrastructure setup" |
| 0.40–0.60 | Related domain | "Military training camp" vs "Army training base" |
| 0.20–0.40 | Loosely related | "Airport construction" vs "Transportation hub development" |
| 0.00–0.20 | Unrelated | "Network infrastructure" vs "Employee payroll" |
| -1.00 | Opposite meaning | Rare in practice with normalized vectors |

**Why cosine similarity works:**
- It measures the angle between two vectors, not the distance
- Two vectors pointing the same direction = similarity 1.0 (regardless of length)
- Two vectors at 90 degrees = similarity 0.0 (orthogonal, unrelated)
- This is robust to text length differences — a long document and a short one can still score 1.0 if they mean the same thing

### Step 4 — Threshold Comparison (Optional)

When a threshold is provided, the system makes a binary decision:

```
if similarity >= threshold:
    match = True   (contextually similar — should inherit weitage)
else:
    match = False  (not similar enough)
```

**How to choose a threshold for your dataset:**

Start with 0.5 and adjust based on results:
- Too many false positives (unrelated docs marked similar)? → raise to 0.6 or 0.7
- Too many false negatives (similar docs missed)? → lower to 0.4

For the military/satellite imagery dataset in `vec_fatboy_data`:
- **0.75+** catches near-duplicates (same event, slightly different wording)
- **0.50** catches docs about the same type of activity
- **0.30** catches loosely related docs in the same domain

---

## End-to-End Flow

### Phase 1: One-Time Batch Embedding (reembed_docs.py)

```
┌─────────────────────────────────────────────────────────────┐
│  Elasticsearch (vec_fatboy_data, 35,000 docs)               │
│  Fields: description, form_title, activity_name, ...        │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       │  Scroll API: fetch 500 docs at a time
                       │  Filter: description exists AND length > 3
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  Sentence Transformer (all-MiniLM-L6-v2)                    │
│  Input:  ["Telecom tower setup...", "Deployment of..."]     │
│  Output: [[0.023, -0.118, ...], [0.031, -0.095, ...]]     │
│          (batch of 500 vectors, each 384-dim)               │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       │  Bulk Update API: 100 docs at a time
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  Elasticsearch (same docs, new field added)                 │
│  New field: desc_embedding = [0.023, -0.118, 0.042, ...]   │
│             (384-dim float array, L2-normalized)            │
└─────────────────────────────────────────────────────────────┘
```

**Performance:** ~16,450 docs with descriptions, processed at ~750 docs/sec = ~22 minutes total.

### Phase 2: Similarity Check (check_two_docs.py)

```
User provides: doc_id_1, doc_id_2, optional threshold
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  Fetch from ES: es.get(index, id=doc_id_1)                  │
│                  es.get(index, id=doc_id_2)                  │
│  Retrieve: desc_embedding field (pre-computed 384-dim vec)  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  Cosine Similarity: dot(vec_1, vec_2)                      │
│  Result: 0.0 to 1.0                                         │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  If threshold provided:                                     │
│    similarity >= threshold → match = True → exit 0          │
│    similarity <  threshold → match = False → exit 1         │
│  Otherwise: just print similarity score                     │
└─────────────────────────────────────────────────────────────┘
```

**Performance:** ~2ms per comparison (just 2 ES fetches + dot product of 384 floats).

### Phase 3: Weitage Propagation (future / in weitage_service.py)

```
For each doc WITH weitage set:
    1. Find all docs where similarity >= threshold
    2. Assign weitage to those docs:
       - Binary mode: same weitage as source
       - Scaled mode: source_weitage * similarity
    3. Write back to ES
```

---

## Two Modes

### Mode 1: By ES Doc ID (`check_by_id`)

- **Input:** Two document IDs from the ES index
- **How:** Fetches pre-computed `desc_embedding` from ES, computes dot product
- **Speed:** ~2ms per comparison (no model loading needed)
- **Requirement:** `desc_embedding` field must exist in ES (run `reembed_docs.py` first)
- **Use when:** You have doc IDs and embeddings are already stored in ES

### Mode 2: By Raw Text (`check_by_text`)

- **Input:** Two raw description strings
- **How:** Loads sentence-transformer model, embeds both texts, computes dot product
- **Speed:** ~50ms per comparison (model load ~3s once, then ~5ms per pair)
- **Requirement:** No ES needed, just the model
- **Use when:** Comparing arbitrary text, testing, or ES doesn't have embeddings yet

---

## Threshold Guide

Based on testing with the `vec_fatboy_data` dataset (military/satellite imagery domain):

| Pair Type | Example | Score |
|-----------|---------|-------|
| Identical text | Same doc vs itself | 1.0000 |
| Near-duplicate | Same event, different wording | 0.95–0.99 |
| Strongly related | "Telecom tower deployment" vs "Network infrastructure" | 0.60–0.85 |
| Related domain | "Military training camp" vs "Army training base" | 0.40–0.60 |
| Loosely related | "Airport" vs "Transportation hub" | 0.20–0.40 |
| Unrelated | "Network" vs "Payroll" | 0.00–0.20 |

**Recommended starting threshold: 0.50**

---

## Files

| File | Purpose |
|------|---------|
| `check_two_docs.py` | Check similarity between two docs. CLI + importable functions. Contains full algorithm docs. |
| `reembed_docs.py` | Batch-embed all ES docs with sentence-transformers and store `desc_embedding` vectors in ES. |
| `requirements.txt` | Python dependencies. |

---

## Setup

```bash
pip install -r requirements.txt
```

**Requirements:**
- Python 3.8+
- Elasticsearch 8.x running at the configured host
- ~200MB disk space for the sentence-transformer model (downloaded on first use)
- For GPU acceleration: CUDA-enabled PyTorch (optional, CPU works fine)

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

**Exit codes:**
- `0` = match (similarity >= threshold) or no threshold provided
- `1` = no match (similarity < threshold)
- `2` = error (missing args, doc not found, etc.)

### Python Import

```python
from check_two_docs import check_by_id, check_by_text

# By ES doc ID (fast, uses pre-computed embeddings)
result = check_by_id("doc_id_1", "doc_id_2", threshold=0.5)
print(result["similarity"])  # 0.8234
print(result["match"])       # True
print(result["description_1"])  # full description text

# By raw text (embeds on-the-fly)
result = check_by_text("Network setup", "Telecom deployment", threshold=0.5)
print(result["similarity"])  # 0.8117
print(result["match"])       # True
```

### Re-embedding ES Docs (Prerequisite for check_by_id)

Before using `check_by_id`, populate the `desc_embedding` field:

```bash
# Full run (all docs)
python reembed_docs.py

# Test with limited docs
python reembed_docs.py --max-docs 100

# Custom batch size
python reembed_docs.py --batch-size 200

# Dry run (no writes)
python reembed_docs.py --dry-run
```

---

## Configuration

ES connection settings are at the top of both scripts:

```
Host:       192.168.1.125
Port:       9200
Scheme:     https
Username:   elastic
Index:      vec_fatboy_data
Field:      desc_embedding
Model:      all-MiniLM-L6-v2
```

To change for your environment, edit the constants at the top of each script.

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `elasticsearch` | >=8.0.0 | Elasticsearch Python client |
| `sentence-transformers` | >=2.2.0 | all-MiniLM-L6-v2 model |
| `numpy` | >=1.21.0 | Vector operations (dot product) |
| `torch` | >=1.10.0 | Backend for sentence-transformers |
