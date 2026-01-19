## Relation-aware Preference Optimization

This directory provides scripts for evaluating trained RPO-RAG models by generating prediction results and computing evaluation metrics.

## Overview

- **Training stage**: Stage 1 (Relation-aware Preference Optimization)
- **Base framework**: SimPO (adapted)
- **Optimization level**: relation-level
- **Output**: intermediate model checkpoints used for downstream fine-tuning

## Setup

1) Install training-specific dependencies:

```bash
pip install -r requirements.txt
```

2) Execute the train script:
```bash
run.sh
```
