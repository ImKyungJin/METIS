# Post-Hoc Merging Is Not Enough: Many-Shot Model Merging with Loss-Gap Balancing

[![ICML 2026](https://img.shields.io/badge/ICML-2026-blue.svg)](https://icml.cc/virtual/2026/poster/64341)
[![arXiv](https://img.shields.io/badge/arXiv-2606.16501-b31b1b.svg)](https://arxiv.org/abs/2606.16501)
[![Project Page](https://img.shields.io/badge/Project-Page-green.svg)](https://ImKyungJin.github.io/METIS)

[Paper](https://arxiv.org/pdf/2606.16501) · [Code](https://github.com/ImKyungJin/METIS) · [ICML Page](https://icml.cc/virtual/2026/poster/64341)

---

## Overview

**METIS** (**M**itigating **E**rasure from **T**ask **I**nterference for **S**table many-shot merging) is a loss-aware many-shot model merging framework designed to mitigate task interference and information erasure in large language models.

Most existing model merging methods rely on **post-hoc one-shot merging**, which aggregates task-specific models in a single step. This often leads to destructive task interference. METIS addresses this by:

1. **Many-Shot Merging**: Iteratively alternating between local task-specific updates and merging over multiple rounds, gradually reducing abrupt parameter shifts.
2. **Loss-Gap-Aware Weighting**: Dynamically rebalancing task contributions at each merging round based on how severely each task's information has been erased.
3. **Consensus-Based Masking**: Filtering out parameter updates dominated by conflicting task contributions, preserving only consistently supported updates.

---

## Environment Setup

### 1. Create and activate a virtual environment

```bash
conda create -n metis python=3.10 -y
conda activate metis
```

### 2. Install dependencies

```bash
pip install -r metis_requirements.txt
```

(Optional) For evaluation:

```bash
pip install -r lmeval_requirements.txt
pip install -r safety-eval_requirements.txt
```

---

## Running METIS

The main entry point is `main_proposed.py`, which uses `fire` for command-line arguments.

```bash
python main_proposed.py run_balanced_model_merging \
  --global_model meta-llama/Llama-3.2-3B \
  --data_path ./data \
  --output_dir ./lora-model_ \
  --num_clients 4 \
  --num_communication_rounds 5
```

All arguments and default values are defined in the `balanced_model_merging()` function in `main_proposed.py`.

---

## Outputs

Merged LoRA adapters are saved under:

```text
lora-model_/<num_clients>/<round>/adapter_model_round<r>.bin
```

The final merged adapter is also saved as:

```text
lora-model_/<num_clients>/adapter_model.bin
```

---

## Evaluation

To evaluate the merged models on multiple benchmarks:

```bash
bash evaluate_all.sh
```

---

## Citation

```bibtex
@inproceedings{im2026posthoc,
  title     = {Post-Hoc Merging is Not Enough: Many-Shot Model Merging with Loss-Gap Balancing},
  author    = {Im, Kyungjin and Kim, Miru and Eom, Chanin and Kwon, Minhae},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  year      = {2026}
}
```
