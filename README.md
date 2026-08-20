# Exact Diameter Computation in Large-Scale Directed Networks

This repository contains the code accompanying the paper:

**Exact Diameter Computation in Large-Scale Directed Networks via Scale-Guided Local Expansion and SCC-DAG Reachability Certification**

The method computes the maximum finite directed shortest-path distance

\[
D=\max_{u\in V}\max_{v:\,d(u,v)<\infty} d(u,v)
\]

on the complete input directed graph. The main algorithm combines sampled sentinel searches, a characteristic distance scale, depth-\(k\) local expansion, SCC decomposition, SCC-DAG reachability-consistent certification (RCC), dynamic lower-bound updates, and exact BFS verification.

## Files

| File | Purpose |
|---|---|
| `rcc_exact_diameter.py` | Main CPU implementation of the proposed exact RCC-based diameter algorithm; produces the results corresponding to the main performance table. |
| `rcc_escape_ablation.py` | Ablation comparing same-SCC certification with full RCC including the SCC-DAG escape certificate; produces the certificate-ablation results. |
| `gpu_exhaustive_diameter.py` | Independent exhaustive CUDA/GPU implementation used to validate the final exact diameter values. |

## Requirements

CPU programs require Python 3 and:

```bash
pip install numpy pandas scipy numba
```

The GPU validation additionally requires:

```bash
pip install torch tqdm
```

with a CUDA-enabled PyTorch installation and an NVIDIA CUDA-capable GPU.

## Input data

Each network must be provided as a whitespace-separated directed edge list:

```text
source_node target_node
```

Lines beginning with `#` are ignored. Node identifiers are remapped internally. Edit the `NETWORK_FILES` list in each script to select the networks to process.

The experiments use the following directed networks:

```text
Web-NotreDame
Web-Stanford
Web-Google
ego-Twitter
Soc-LiveJournal1
Soc-Pokec
ego-Gplus
Wiki-Talk
```

## Reproducibility

For the CPU experiments:

- sentinel sample size: `500`
- random seed: `2026`
- maximum worker processes: `14`
- local expansion depth:
  \[
  k=\max\left(1,\left\lfloor\hat L/2\right\rfloor\right)
  \]

The main algorithm returns the exact diameter because local scale information is used only for search guidance, while pruning is based on rigorous RCC bounds and safe early termination; all remaining candidates are evaluated by exact BFS.

The GPU program is algorithmically independent of RCC and performs exhaustive source-based validation.

## Suggested execution order

```bash
python rcc_exact_diameter.py
python rcc_escape_ablation.py
python gpu_exhaustive_diameter.py
```

The first script reproduces the main RCC results, the second reproduces the escape-certificate ablation, and the third independently checks the reported diameter values.
