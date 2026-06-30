# Research / Literature Review

This directory contains 39 structured notes on papers we reviewed during the design of tiny-cl's three mechanisms (SLAO, MVA, AVR). Files are in two flavors:

- `*_search.json` — lightweight metadata (title, authors, abstract, arXiv ID, GitHub repo if any)
- `*_arxiv.json` — full paper text extracted from arXiv HTML, used for equation-level reading

## Index by topic

### Self-improvement & self-rewarding (MVA lineage)

| File                         | Paper                                          | Why we read it                          |
|------------------------------|------------------------------------------------|-----------------------------------------|
| `self_refine_*`              | Self-Refine (Madaan et al., NeurIPS 2023)      | Iterative self-correction without external reward |
| `self_notes_search.json`     | Self-Rewarding LMs follow-ups                  | Survey of self-reward mechanisms        |
| `intuitor_arxiv.json`        | INTUITOR / RLIF (Zhao et al., ICLR 2026)       | **Direct inspiration for MVA's certainty signal** (`KL(U‖p)`) |
| `si_no_reward_search.json`   | "Self-Improvement Without External Reward"     | Verifies the no-reward setting          |
| `intrinsic_reward_search.json` | Intrinsic-reward survey                       | Comparison point for certainty as intrinsic reward |

### Continual learning & merging (SLAO lineage)

| File                         | Paper                                          | Why we read it                          |
|------------------------------|------------------------------------------------|-----------------------------------------|
| `seal_*`                     | SEAL (Zweiger et al., NeurIPS 2025)            | Self-editing continual learning; comparison point |
| `azr_details_search.json`    | Absolute-Zero-Reasoner details                 | Self-play RL without external data      |
| `absolute_zero_*`            | AZR paper (Yu et al., 2025)                    | Self-play on code/math                  |
| `agent_r_*`                  | Agent-R (Self-rewarding agent)                 | Multi-step agent self-improvement       |

### Test-time training & in-context adaptation

| File                         | Paper                                          | Why we read it                          |
|------------------------------|------------------------------------------------|-----------------------------------------|
| `vds_ttt_*`                  | VDS-TTT                                        | Test-time training with variance reduction |
| `in_place_ttt_*`             | In-place TTT (Wang et al.)                     | Memory-efficient TTT (relevant to on-device) |
| `ttt_search.json`            | TTT survey                                     | Background                              |

### Sleep / offline consolidation

| File                         | Paper                                          | Why we read it                          |
|------------------------------|------------------------------------------------|-----------------------------------------|
| `lm_sleep_*`                 | LM-Sleep (Wang et al.)                         | Offline consolidation via generated dreams |
| `behrouz_sleep_*`            | Behrouz et al. sleep variant                   | Alternative consolidation mechanism     |

### Reasoning & search

| File                         | Paper                                          | Why we read it                          |
|------------------------------|------------------------------------------------|-----------------------------------------|
| `rstar_math_*`               | rStar-Math (Xia et al.)                        | Self-rewarding math reasoning           |
| `r_zero_*`                   | R-Zero                                         | RL without external reward, code        |
| `recursive_thinking_search.json` | Recursive thinking                         | Multi-step reasoning without tool calls |
| `rest_em_*`                  | ReST-EM (Singh et al.)                         | Self-improvement via rejection-sampling + EM |
| `rest_mcts_search.json`      | ReST-MCTS                                      | Search-augmented self-improvement       |
| `ttrl_*`                     | Test-Time Reinforcement Learning               | RL at inference time                    |
| `sqlm_*`                     | SQLM (Self-Quality Language Model)             | Quality-graded self-improvement         |
| `prm_search.json`            | Process Reward Models                          | Step-level reward comparison            |

### Other / context

| File                         | Paper                                          | Why we read it                          |
|------------------------------|------------------------------------------------|-----------------------------------------|
| `crisp_search.json`          | CRISP                                          | Continual learning baseline comparison  |
| `iclr2026_search.json`       | ICLR 2026 paper list                           | Where SLAO and INTUITOR appeared        |
| `rlef_search.json`           | RLEF                                           | RL from environment feedback            |
| `srl_search.json`            | SRL (Self-Referential Learning)                | Self-referential mechanism              |
| `ondevice_search.json`       | On-device LLM survey                           | Hardware constraints for our setting    |
| `trt_*`                      | TRT (Transformer Recursive Training)           | Comparison point                        |

## How to use these notes

Each `*_arxiv.json` is the full text of the paper, scraped from arxiv.org. Use them to:

- Verify our equations match the original (e.g., does our `KL(U‖p)` match INTUITOR's definition?)
- Pull exact numbers for comparison (e.g., SEAL's continual-learning accuracy)
- Trace the citation graph (which papers cite which)

The `*_search.json` files are lightweight — use them as an index to find papers by topic without loading full text.

## Notes on the corpus

- We focused on **post-2024 papers** because the field shifted dramatically with INTUITOR (May 2025) and SLAO (Dec 2025).
- We did **not** include the seminal 2019–2022 continual-learning literature (EWC, LwF, etc.) because those are well-surveyed elsewhere and our V2 framework already implements them as baselines.
- We **did** include papers that turned out to be wrong or superseded (e.g., the "SEA-RAFT" confusion, certain certainty-based RLIF critiques) because the negative results informed our design.
