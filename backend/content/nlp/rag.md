---
title: "Retrieval-Augmented Generation (RAG)"
slug: rag
summary: "Combining LLMs with external knowledge retrieval to reduce hallucinations and improve factual accuracy in AI-generated responses."
tags: ["RAG", "retrieval", "embeddings", "generation", "LLM", "vector-search"]
visibility: public
---

# Retrieval-Augmented Generation (RAG)

## Overview

Large Language Models have a fixed knowledge cutoff and can hallucinate facts. **Retrieval-Augmented Generation (RAG)** addresses this by coupling a retriever with a generator — the model retrieves relevant documents from an external knowledge base and uses them as context when generating answers.

**Paper:** "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Lewis et al., 2020)

**Formula:** Given a query $q$ and a corpus of documents $\mathcal{D}$:

$$p(y | x) = \sum_{z \in \text{top-}k} p_\eta(z | x) \cdot p_\theta(y | x, z)$$

Where:
- $p_\eta(z | x)$ — retriever probability (dense passage retrieval)
- $p_\theta(y | x, z)$ — generator probability conditioned on context
- $z$ — retrieved document passage

---

## RAG vs Fine-Tuning

| Dimension | RAG | Fine-Tuning |
|-----------|-----|-------------|
| Knowledge update | Real-time (update index) | Requires retraining |
| Hallucination | Reduced (grounded) | Still possible |
| Cost | Index + inference | Training cost |
| Interpretability | Can cite sources | Opaque |
| Dynamic knowledge | ✅ Excellent | ❌ Static |

---

## Architecture: Three-Phase Pipeline

### Phase 1: Indexing (Offline)

```
Documents → Chunking → Embedding → Vector Store
```

**Document Loading:**
- PDFs, web pages, databases, APIs, markdown files
- Chunking strategy matters: too small = no context; too large = diluted

**Chunking Strategies:**

| Strategy | Method | Chunk Size |
|----------|--------|------------|
| Fixed-size | Split every N tokens | 256–512 tokens |
| Sentence splitter | Split at sentence boundaries | Variable |
| Semantic | Group semantically similar sentences | Variable |
| Recursive | Hierarchical splitting | Adaptive |

**Embedding:**
Each chunk $d_i$ is encoded into a dense vector:
$$\mathbf{v}_i = \text{Encoder}(d_i) \in \mathbb{R}^d$$

Common embedding models:
- `text-embedding-3-large` (OpenAI) — 3072 dims
- `BGE-M3` (BAAI) — multilingual
- `E5-large` — strong zero-shot retrieval
- `all-MiniLM-L6-v2` (sentence-transformers) — fast, compact

**Vector Store:**
Stores embeddings + metadata for fast approximate nearest neighbor (ANN) search.

Popular stores: Pinecone, Weaviate, Chroma, Qdrant, FAISS, pgvector

### Phase 2: Retrieval (Query Time)

```
User Query → Query Embedding → ANN Search → Top-k Chunks
```

**Dense Retrieval (semantic):**
$$\text{sim}(q, d) = \cos(\mathbf{v}_q, \mathbf{v}_d) = \frac{\mathbf{v}_q \cdot \mathbf{v}_d}{\|\mathbf{v}_q\| \|\mathbf{v}_d\|}$$

**Sparse Retrieval (BM25):**
$$\text{BM25}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t,d) \cdot (k_1 + 1)}{f(t,d) + k_1 \cdot (1 - b + b \cdot |d|/\text{avgdl})}$$

Where $f(t,d)$ is term frequency, $k_1 = 1.2$, $b = 0.75$.

**Hybrid Search:**
Combines dense + sparse scores:
$$\text{score}(q, d) = \alpha \cdot \text{dense}(q, d) + (1 - \alpha) \cdot \text{sparse}(q, d)$$

### Phase 3: Generation (Augmented Inference)

```
[System Prompt + Retrieved Chunks + User Query] → LLM → Response
```

**Prompt Template:**
```
System: You are a helpful assistant. Answer ONLY based on the provided context.
If the answer is not in the context, say "I don't know."

Context:
{chunk_1}
{chunk_2}
{chunk_3}

User: {question}
Assistant:
```

---

## Retrieval Quality

### Metrics

| Metric | Measures | Formula |
|--------|----------|---------|
| **Hit Rate@k** | Relevant doc in top-k | $\frac{\text{queries with hit}}{N}$ |
| **MRR@k** | Rank of first relevant doc | $\frac{1}{N}\sum \frac{1}{\text{rank}_i}$ |
| **NDCG@k** | Graded relevance, position-aware | See RecSys metrics |
| **Recall@k** | Fraction of relevant docs retrieved | $\frac{|\text{relevant} \cap \text{retrieved}|}{|\text{relevant}|}$ |

---

## Advanced RAG Techniques

### 1. Query Transformations

**HyDE (Hypothetical Document Embeddings):**
Generate a hypothetical answer, embed it, retrieve similar real documents:
```
q → LLM → hypothetical_answer → embed → retrieve
```

**Multi-query retrieval:**
```python
queries = llm.generate([
    "What is X?",
    "How does X work?",
    "Applications of X?"
])
results = [retriever.retrieve(q) for q in queries]
merged = deduplicate(results)
```

**Step-back prompting:**
Abstract the question before retrieving: "What is the underlying principle of...?"

### 2. Reranking

After retrieving top-k (e.g., k=20), rerank with a cross-encoder for top-m (m=5):

```
Query + Doc → Cross-Encoder → Relevance Score → Re-sort
```

Cross-encoders (Cohere Rerank, BGE-reranker, Jina Reranker) are slower but more accurate since they see query and doc jointly.

$$\text{score}(q, d) = \text{CrossEncoder}([q; d])$$

### 3. Contextual Compression

Extract only relevant parts of retrieved chunks:
```python
# Don't pass full chunk — extract relevant sentences
relevant_sentences = llm.extract_relevant(chunk, query)
```

### 4. Hierarchical RAG

Two-level retrieval:
1. **Coarse**: Retrieve relevant sections/documents
2. **Fine**: Retrieve specific passages within those documents

Reduces noise while maintaining context.

### 5. Self-RAG

LLM decides **when** to retrieve and **whether** to use retrieved content:

1. Generate → check if retrieval needed → retrieve if yes
2. Generate with context → critique relevance → keep or discard
3. Final output with reflection tokens (`[Retrieve]`, `[IsRel]`, `[IsSup]`)

---

## RAG Failure Modes

| Problem | Cause | Fix |
|---------|-------|-----|
| **Missing context** | Relevant doc not retrieved | Increase k, better embeddings, hybrid search |
| **Wrong chunk** | Poor chunking strategy | Semantic chunking, overlap |
| **Lost in middle** | LLM ignores middle context | Reranking, reduce context length |
| **Conflicting docs** | Sources disagree | Citation, confidence scoring |
| **Faithfulness** | Model ignores context | Stricter prompting, fine-tuning |

---

## Evaluation Framework

**RAGAs (Retrieval-Augmented Generation Assessment):**

| Metric | What it Measures |
|--------|-----------------|
| **Context Precision** | Retrieved chunks actually relevant |
| **Context Recall** | All needed info was retrieved |
| **Faithfulness** | Answer is grounded in context |
| **Answer Relevancy** | Answer addresses the question |

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision]
)
```

---

## Implementation Stack

```python
# LangChain RAG pipeline
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# 1. Embed and index documents
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=OpenAIEmbeddings()
)

# 2. Create retriever (top-5 chunks)
retriever = vectorstore.as_retriever(
    search_type="mmr",    # Maximum Marginal Relevance
    search_kwargs={"k": 5, "fetch_k": 20}
)

# 3. RAG chain
chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4"),
    retriever=retriever,
    return_source_documents=True
)

response = chain.invoke({"query": "What is attention mechanism?"})
```

**MMR (Maximum Marginal Relevance)** balances relevance and diversity:
$$\text{MMR} = \arg\max_{d \in R \setminus S} [\lambda \cdot \text{sim}(d, q) - (1-\lambda) \max_{s \in S} \text{sim}(d, s)]$$

---

## Key Takeaways

1. **RAG = Retriever + Generator** — retrieve relevant docs, generate grounded answers
2. **Three phases:** Offline indexing → query-time retrieval → augmented generation
3. **Hybrid search** (dense + sparse) consistently outperforms either alone
4. **Chunking strategy** is often the biggest lever for retrieval quality
5. **Reranking** (cross-encoder) significantly improves precision at small cost
6. **Evaluate with RAGAs** — measure retrieval, faithfulness, and answer quality separately
7. **Advanced RAG** (HyDE, self-RAG, step-back) improves complex queries

## References

- Lewis et al. (2020) — Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
- Gao et al. (2023) — Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE)
- Asai et al. (2023) — Self-RAG: Learning to Retrieve, Generate, and Critique
- Es et al. (2023) — RAGAS: Automated Evaluation of Retrieval Augmented Generation
