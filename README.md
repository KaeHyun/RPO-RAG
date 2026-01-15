## RPO-RAG (WWW 2026)
This repository is the official implementation of **RPO-RAG: Aligning Small LLMs with Relation-aware Preference Optimization for Knowledge Graph Question Answering**.
<p align="center">
  <img src="https://img.shields.io/badge/Task-KGQA-blue" />
  <img src="https://img.shields.io/github/last-commit/KaeHyun/RPO-RAG" />
  <img src="https://img.shields.io/github/stars/KaeHyun/RPO-RAG" />
</p>

---
## 🧠 Why RPO-RAG ?
<table align="center">
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/59ebacd2-533d-43fe-97fb-10d2753da2fc" width="450" /><br/>
      <sub>Baseline Prompt</sub>
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/4e4b1714-73ec-4c06-952c-e454dd6d2e9e" width="450" /><br/>
      <sub>Answer-Centered Prompt (Ours)</sub>
    </td>
  </tr>
</table>

Large Language Models (LLMs) have shown strong reasoning abilities, but they frequently hallucinate on knowledge-intensive tasks.
KG-based Retrieval-Augmented Generation (KG-RAG) mitigates this issue by grounding answers in structured knowledge graphs.

However, existing KG-RAG methods suffer from three critical limitations:

- **Semantics-unaware path sampling**  
  Reasoning paths are often selected using shortest-path heuristics, which are topologically close but semantically irrelevant to the query.

- **Weak alignment with KG reasoning objectives**  
  Training focuses on predicting final answers, while intermediate relational reasoning is largely unsupervised.

- **Flat prompting that hinders small LLMs**  
  Retrieved paths are presented as unordered lists, making it difficult for small models (≤8B) to integrate evidence coherently.

These issues disproportionately affect **small LLMs**, as larger models can partially compensate for noisy retrieval using extensive parametric knowledge.
RPO-RAG is designed to address these challenges by aligning retrieval, optimization, and prompting with the relational structure of knowledge graphs.

## 🧩 Overall Framework
<p align="center">
  <img src="https://github.com/user-attachments/assets/61f62e8a-b053-4846-a31b-520d1e931425" width="800" />
</p>

RPO-RAG follows a unified retrieval–reasoning pipeline explicitly aligned with knowledge graph semantics.
The framework consists of three main components:

**1️⃣ Query-Path Semantic Sampling**  
Constructs query-aligned reasoning paths via semantic clustering, providing high-quality supervision for both retrieval and reasoning.

**2️⃣ Semantic-Matching Retriever**  
Retrieves reasoning paths that are semantically consistent with the query using a lightweight pretrained language model.

**3️⃣ Dual-Objective Optimization**  
Jointly optimizes relation-aware preference learning and answer-centered prompt design to align small LLMs with structured KG reasoning.

## Quick Start 

## 🚀 How to Use

### 1️⃣ Install Requirements
Install the required dependencies:

```bash
pip install -r requirements.txt

## 2️⃣ Train Model

RPO-RAG is trained via **two-stage optimization**, corresponding to the two core objectives described in the paper.
Each stage is implemented in a separate directory.

---
### (1) Relation-aware Preference Optimization

Relation-aware preference optimization is implemented by **modifying the SimPO framework**.
We adapt SimPO to operate at the **relation level** for knowledge graph reasoning.

- **Codebase**: adapted from *SimPO*
- **Directory**: `./train_RPO/relation_aware_preference_optimization`

Please refer to the scripts and configurations inside this directory for training details.

---
### (2) Task-aware Optimization (Answer-Centered Prompting)

Task-aware optimization is performed via **instruction fine-tuning** using models and utilities released by **Unsloth**.

- **Base framework**: *Unsloth*
- **Objective**: instruction fine-tuning for answer-centered reasoning
- **Directory**: `./train_RPO/tast_aware_optimization`

The training scripts and configurations are provided in the corresponding folder.
