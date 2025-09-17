# Amazon Scout

<p align="center"><a href="https://amazon-scout.vercel.app/">amazon-scout.vercel.app</a></p>

<p align="center">
  <img src="docs/banner.png" alt="Qdrant Hybrid Search — phone → Qdrant → Hybrid → Filters → Rerank → Results" width="85%"/>
</p>

<p align="center">
  <a href="https://img.shields.io/badge/Python-3.10%2B-blue.svg"><img src="https://img.shields.io/badge/Python-3.10%2B-blue.svg" /></a>
  <a href="https://img.shields.io/badge/Qdrant-Cloud-success"><img src="https://img.shields.io/badge/Qdrant-Cloud-success" /></a>
  <a href="https://img.shields.io/badge/GCP-Cloud%20Run%20%7C%20GKE-orange"><img src="https://img.shields.io/badge/GCP-Cloud%20Run%20%7C%20GKE-orange" /></a>
  <a href="https://img.shields.io/badge/License-MIT-green"><img src="https://img.shields.io/badge/License-MIT-green" /></a>
</p>

> **TL;DR** — Production-ready **hybrid search** on **2M Amazon reviews** (Dense + BM25) with **facets**, **reranking**, and **quantization** on **Qdrant Cloud (GCP)**.
> ✅ P\@10 ≈ **99.8–100%** on our eval set · 💾 **\~30% storage** saved with INT8 scalar quantization · 🌎 globally reachable API.

---

## 📌 Features

* **Hybrid retrieval**: Dense + BM25 (sparse) fused for high recall
* **Facet filters**: price, rating, category, brand, verified purchase, date, etc.
* **Rerank stage**: plug-in *MiniCOIL* / *ColBERT* / *Cohere Rerank* (choose your flavor)
* **Quantization**: optional INT8 / PQ for lower storage & faster I/O
* **Cloud-native**: Qdrant Cloud on GCP + FastAPI microservice (ready for Cloud Run)
* **Batched ingestion**: stream JSONL/Parquet and index at scale
* **Eval kit**: Precision\@K / Recall\@K / NDCG
* **🤖 GPT Recommendations Agent (human‑review grounded)**: an LLM layer that **reads the retrieved human reviews** (not ads), summarizes pros/cons, and **recommends products** with **quoted evidence** from real user text. No sponsored placement, no brand bias—**recommendations are grounded in what people actually said.**

---

## 🧭 Architecture

The system is intentionally simple so it’s easy to operate and extend.

> **Why this order?** Hybrid gets you **recall**. Filters keep it **relevant**. Rerank polishes **top‑K**. Kid-simple: *find a big pile → keep only what you asked for → sort the best on top.*

### + Agent layer for trusted recommendations

**What the agent does**

* Consumes the **top‑K** retrieved review payloads (title, text, stars, price, brand, etc.).
* Produces **concise, human‑readable recommendations** (e.g., Top 3) with **grounded snippets** from real reviews.
* **Cites** each reason with `review_id` + quoted text so readers can verify.
* Enforces **“facts‑from‑reviews‑only”**: avoids hallucinations and marketing copy.

---

## 🚀 Quickstart (No-Code Overview)

1. **Create a Python environment** and install: Qdrant Client (with FastEmbed), FastAPI, uvicorn, orjson, pydantic. Optionally add sentence-transformers (for rerankers) and pandas/pyarrow (for Parquet).
2. **Configure Qdrant Cloud**: set your endpoint URL and API key as environment variables. Use Qdrant Cloud (GCP) project/cluster.
3. **Create the collection** with two vector families: a 384‑dim **dense** vector (cosine) and a **bm25** sparse vector. Define typed payload fields for facets (price, stars, category, brand, verified, date…).
4. **Ingest data** from JSONL/Parquet: embed `title + text` for dense and sparse, upsert with payload. Use batching (e.g., 512) for throughput.
5. **(Optional) Quantize** dense vectors (INT8 or PQ) after validating quality. Expect \~30% storage savings with scalar INT8.
6. **Serve a search API** (e.g., FastAPI) that performs **Hybrid → Filters → Rerank** and returns review cards. Deploy on **Cloud Run**.

---

## 🤖 GPT Recommendations Agent (Human‑Trusted Reviews → Product Picks)

**Mission:** *Recommend products using only what real people wrote.* Not ads, not manufacturer blurbs—**ground everything** in the retrieved review texts.

### What you get

* **Summarized pros/cons** per product with **evidence quotes** from actual reviews
* **Top‑N picks** tailored to the query and facet filters
* Optional **Q\&A** over reviews (e.g., “Is this noisy?”, “Fits wide feet?”) with citations

### Why trust this?

* **No sponsored placement** and **no brand bias** — everything flows from what reviewers said.
* **Verifiable**: Every claim includes a `review_id` + **quoted snippet** so readers can check the source.
* **Guardrails**: The agent is instructed to avoid speculation or unverifiable claims.

### How to enable

* Provide credentials for your preferred LLM provider (OpenAI/Anthropic/Vertex/other).
* Keep temperature low (≈0.1–0.3) for stable, audit‑friendly outputs.
* The agent consumes only the retrieved review payloads and returns concise JSON with `{id, title, why, quotes[]}` — where `quotes[]` are short verbatim snippets with `review_id`s.

### Endpoints (conceptual)

* **`/search`** — Hybrid search with facet filters; returns candidate review cards.
* **`/recommend`** — Runs `/search`, then the GPT agent to produce Top‑N picks with quotes.
* **`/ask`** — Free‑form Q\&A over the retrieved review set; answers include citations.

---

## 📊 Evaluation

* Report **Precision\@K / Recall\@K / NDCG** on a curated test set. We observed **P\@10 ≈ 99.8–100%** after reranking (dependent on model choices and label quality).
* Agent-side health: track **evidence coverage** (% of recommendation sentences backed by quotes) and **contradiction rate** (whether quotes conflict with the claim).

---

## 🧩 Project Layout (Conceptual)

* `api.py` — FastAPI service surface (search + recommend + ask)
* `data/reviews.jsonl` — Input dataset (id/title/text/stars/price/...)
* `docs/` — Visuals (banner, diagrams)
* `scripts/ingest.py` — Batch ingestion/indexing logic
* `requirements.txt` — Python deps
* `README.md` — This file

---

## 🧱 Design notes (teammate truth)

* **Hybrid first** for **recall**, **filters second** for **precision**, **rerank last** for **quality**. That order is correct — *kid-simple pipeline*.
* **Agent** only reads **what was retrieved**; it **quotes** evidence for every claim and avoids marketing language.
* **Quantize** only after you’ve validated quality (always run A/B on P\@K).
* Keep the **payload** skinny and **facets** typed (numeric for ranges, enums for categories) for fast filtering.
* Oversample (e.g., `top_k × 5`) **before** rerank; otherwise the reranker can’t help enough.

---

## 📝 License

MIT — do good things. If this saves you time, ⭐ the repo and tell a friend.

---

## 🙌 Acknowledgements

* Built with **Qdrant Cloud** on **Google Cloud Platform**
* Dense + Sparse via **FastEmbed**; bring your own models if you prefer
* Optional rerankers: **MiniCOIL / ColBERT / Cohere Rerank**
* GPT Recommendations Agent: generic LLM interface (plug your provider of choice).
