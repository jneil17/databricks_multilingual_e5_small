# multilingual-e5-small on Databricks

Notebooks for deploying and fine-tuning the [multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small) text embedding model on Databricks using MLflow, Unity Catalog, and Model Serving.

| Notebook | Description |
|----------|-------------|
| [`deploy_multilingual_e5_small.ipynb`](deploy_multilingual_e5_small.ipynb) | Deploy the base model to a serving endpoint |
| [`finetune_multilingual_e5_small.ipynb`](finetune_multilingual_e5_small.ipynb) | Fine-tune on your own data using CPU (no GPU required) |

## What is multilingual-e5-small?

multilingual-e5-small is a lightweight, multilingual text embedding model from Microsoft Research, published in [ACL 2024](https://arxiv.org/abs/2402.05672). It maps text into 384-dimensional dense vectors for semantic similarity tasks.

- **Core function**: Generate text embeddings for semantic search, retrieval, classification, and clustering
- **Training**: Weakly-supervised contrastive pre-training on 1 billion multilingual text pairs, fine-tuned on labeled datasets
- **Languages**: 100+ languages
- **Size**: ~470MB (~134M parameters)
- **Max sequence length**: 512 tokens
- **Output dimensions**: 384

### Input Prefixing

For best results, prefix your inputs:
- **Queries** (questions, search terms): `"query: How do I reset my password?"`
- **Passages** (documents, paragraphs to index): `"passage: To reset your password, navigate to Settings..."`

This asymmetric prefixing is part of the model's training and significantly improves retrieval quality.

### Use Cases

- Semantic search and document retrieval
- Text classification and clustering
- Retrieval-Augmented Generation (RAG) pipelines
- Cross-lingual similarity matching
- Duplicate detection

## Prerequisites

### Infrastructure

- **Databricks workspace** with Unity Catalog enabled
- **ML cluster** for model download and registration
  - Any ML-capable cluster works (no GPU required for download)
  - Databricks Runtime: **15.4 LTS ML** or later
- **Databricks Volumes** for model storage (~500MB)

### Software Dependencies

- `transformers`
- `torch`
- `sentence-transformers`
- `mlflow` (with Unity Catalog)
- Python 3.10+

## Deployment Process

The included notebook walks through these steps:

| Step | Description | Time Estimate |
|------|-------------|---------------|
| 1 | Create an ML cluster | 3-5 min |
| 2 | Install dependencies (`transformers`, `torch`, `sentence-transformers`) | 2-3 min |
| 3 | Configure Unity Catalog storage (catalog, schema, volume) | 1 min |
| 4 | Download and register the model via MLflow with `llm/v1/embeddings` task | 5-10 min |
| 5 | Create a serving endpoint | 5 min + ~10 min warmup |

**Total: ~25-35 minutes**

### Critical: Task Type

The model **must** be registered with task type `llm/v1/embeddings`. This enables the Databricks embeddings API format.

### Serving Options

Because the model is small (~470MB), you have flexible serving options:
- **Serverless** (recommended): No infrastructure management, auto-scales to zero
- **Provisioned Throughput**: Dedicated resources for predictable latency

## Quick Start

1. Import `deploy_multilingual_e5_small.ipynb` into your Databricks workspace
2. Create an ML cluster (standard ML runtime, no special GPU requirements)
3. Update the configuration variables:
   - `CATALOG_NAME` — your Unity Catalog catalog
   - `SCHEMA_NAME` — your schema
4. Run all cells (~10 minutes)
5. Create a serving endpoint via the Databricks UI

## Testing the Endpoint

### Via Python API

```python
from openai import OpenAI
import os

DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN")

client = OpenAI(
    api_key=DATABRICKS_TOKEN,
    base_url="https://<your-workspace>.cloud.databricks.com/serving-endpoints"
)

response = client.embeddings.create(
    model="multilingual-e5-small-embeddings",
    input=[
        "query: What is Databricks?",
        "passage: Databricks is a unified analytics platform for data engineering and data science."
    ]
)

for i, embedding in enumerate(response.data):
    print(f"Input {i}: {len(embedding.embedding)} dimensions")
    print(f"  First 5 values: {embedding.embedding[:5]}")
```

### Via cURL

```bash
curl -X POST "https://<your-workspace>.cloud.databricks.com/serving-endpoints/multilingual-e5-small-embeddings/invocations" \
  -H "Authorization: Bearer $DATABRICKS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "input": [
      "query: What is Databricks?",
      "passage: Databricks is a unified analytics platform."
    ]
  }'
```

### Computing Similarity

```python
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

query_embedding = response.data[0].embedding
passage_embedding = response.data[1].embedding

similarity = cosine_similarity(query_embedding, passage_embedding)
print(f"Similarity: {similarity:.4f}")
```

## Fine-Tuning on CPU

The `finetune_multilingual_e5_small.ipynb` notebook walks through fine-tuning the model on your own domain data using **CPU-only** compute. This is useful when GPU instances are unavailable (e.g., GCP availability constraints).

### Requirements
- GCP instance: `n2-standard-16` (64GB RAM) or `n2-standard-32` (128GB RAM)
- Databricks Runtime: 15.4 LTS ML or later
- Training data as query/passage pairs with similarity scores in a Delta table

### Training Time Estimates (CPU)
| Dataset Size | Epochs | Approximate Time |
|-------------|--------|-----------------|
| 100 pairs | 3 | < 1 minute |
| 1,000 pairs | 3 | 15-30 minutes |
| 10,000 pairs | 3 | 2-4 hours |

### GPU Upgrade Path
When GPU instances become available, change `device="cpu"` to `device="cuda"` and set `fp16=True` in training arguments. Everything else stays the same.

## Deployment Architecture

```
User Request (text strings)
    |
Databricks Serving Endpoint (Serverless or Provisioned Throughput)
    |
MLflow Model Registry (Unity Catalog)
    |
multilingual-e5-small (~470MB in Volumes)
    |
Response (384-dimensional embeddings)
```

## References

- [multilingual-e5-small on Hugging Face](https://huggingface.co/intfloat/multilingual-e5-small)
- [Multilingual E5 Paper (arXiv)](https://arxiv.org/abs/2402.05672)
- [Databricks Model Serving Docs](https://docs.databricks.com/en/machine-learning/model-serving/index.html)
- [MLflow Sentence Transformers Integration](https://mlflow.org/docs/latest/llms/sentence-transformers/index.html)

## License

This deployment guide is provided as-is. The multilingual-e5-small model is released under the [MIT License](https://huggingface.co/intfloat/multilingual-e5-small/blob/main/LICENSE).
