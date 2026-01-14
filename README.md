## RPO-RAG (WWW 2026)
This repository is the official implementation of **RPO-RAG: Aligning Small LLMs with Relation-aware Preference Optimization for Knowledge Graph Question Answering**.
<p align="center">
  <img src="https://img.shields.io/badge/Task-KGQA-blue" />
  <img src="https://img.shields.io/github/last-commit/KaeHyun/RPO-RAG" />
  <img src="https://img.shields.io/github/stars/KaeHyun/RPO-RAG" />
</p>

---
## 🧠 Why RPO-RAG ?
<p align="center">
  <img src="https://github.com/user-attachments/assets/1492a3eb-8dd3-416b-8418-5dee8898fecc" width="600" />
</p>
Large Language Models (LLMs) have shown strong reasoning abilities, but they frequently hallucinate on knowledge-intensive tasks.
KG-based Retrieval-Augmented Generation (KG-RAG) mitigates this issue by grounding answers in structured knowledge graphs.

However, existing KG-RAG methods suffer from three critical limitations:

- **Semantics-unaware path sampling**  
  Reasoning paths are often selected using shortest-path heuristics, which are topologically close but semantically irrelevant to the query.

- **Weak alignment with KG reasoning objectives**  
  Training focuses on predicting final answers, while intermediate relational reasoning is largely unsupervised.

- **Flat prompting that hinders small LLMs**  
  Retrieved paths are presented as unordered lists, making it difficult for small models (≤8B) to integrate evidence coherently.

These issues disproportionately affect **small LLMs**, which are far more sensitive to noisy supervision and poorly structured evidence.
RPO-RAG is designed to address these challenges by aligning retrieval, optimization, and prompting with the relational structure of knowledge graphs.

## 🧩 Overall Framework
<p align="center">
  <img src="https://github.com/user-attachments/assets/61f62e8a-b053-4846-a31b-520d1e931425" width="800" />
</p>

RPO-RAG follows a unified retrieval–reasoning pipeline explicitly aligned with knowledge graph semantics.
The framework consists of three main components:

**1️⃣ Query-Path Semantic Sampling**

**2️⃣ Relation-aware Preference Optimization**

**3️⃣ Answer-Centered Prompting**
