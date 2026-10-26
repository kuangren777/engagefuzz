# EngageFuzz

EngageFuzz is a coverage-guided fuzzer for indirect prompt injection against tool-using LLM agents.
It scores each attack execution on a staged proximity ladder (did the agent see the injection, call a
side-effect tool, pass an attacker-unique argument, complete the goal) and uses that score as the
fuzzing reward. This repository holds the engine, the analysis pipeline and the frozen result data
behind every table, figure and macro in the paper.

## Layout

| Path | Content |
|---|---|
| `engine/` | Campaign loop, reward arms (`random`, `binary`, `coverage`, `prox-v21`, `tmap-ladder`, ...), census and target-selection runners, unit tests |
| `testbed/` | AgentDojo harness (`harness.py`), mutation operators (`fuzzer.py`), design/frontier data in `testbed/data/` |
| `analysis/` | Statistics, tables, figures and LaTeX macros, `reproduce_all.sh` runs them all |
| `paper_static/` | Paper tables whose cells are all macros (layout only) |
| `injecagent/` | RQ4 transfer to InjecAgent (runner and proximity grader) |
| `runs/` | Frozen run data: `rq1/` (census), `rq2/` (campaigns), `rq4/` (InjecAgent), `select/targets_qwen3-8b-local.json` |
| `plan.md`, `NOTES_errata.md` | Pre-registration log and errata |

## Dependencies

- Python 3.10+, `scipy`, `numpy`, `matplotlib`, `openai`, `nltk` (InjecAgent parsing).
- External benchmarks, not vendored. Install them under their own licenses.
  - AgentDojo v1.2: https://github.com/ethz-spylab/agentdojo . Either `pip install agentdojo` or `export AGENTDOJO_SRC=/path/to/agentdojo/src`.
  - InjecAgent: https://github.com/uiuc-kang-lab/InjecAgent . `export INJECAGENT_DIR=/path/to/InjecAgent`.
- Open models are served with vLLM behind an OpenAI-compatible endpoint, one port per model (ports are registered in `testbed/harness.py` `LOCAL_PORTS`, `--port` overrides per process).
- Optional defenses (`tripwire`, `melon`, `pi_detector`) load external code. Set `TRIPWIRE_DIR` and `PIDET_MODEL` if you use them. They are not needed for the paper tables.

## Credentials

API models (the LLM judge of the `judge-ladder` and `tmap-ladder` arms, the frontier models of RQ3/RQ4) go through an
OpenAI-compatible gateway that you choose. Set both variables yourself. There is no default host and no key in the code.

```bash
export LLM_GATEWAY_URL=https://your-gateway.example.com   # without the trailing /v1
export LLM_GATEWAY_KEY=...                                # your key
export EF_JUDGE=gpt-5.5                                 # judge model name on your gateway (optional)
```

They can also be placed in `~/.config/engagefuzz/env` as `KEY=value` lines. Runs that use only local vLLM models never call the gateway.
Analysis from the frozen data needs neither the gateway nor a GPU, only `AGENTDOJO_SRC` (the analysis imports AgentDojo task suites).

## Reproduce the paper numbers from `runs/`

One command regenerates everything into `out/` and, given a checkout of the paper sources, compares each file with the paper copy:

```bash
export AGENTDOJO_SRC=/path/to/agentdojo/src      # skip if agentdojo is pip-installed
bash analysis/reproduce_all.sh [/path/to/paper]
```

Every `\input` file of the paper maps to one generator. All `.tex` outputs are byte-identical to the paper; the figure PDFs are identical except for the embedded creation date and document ID.

| Paper file | Generator (in `analysis/`) | Data read |
|---|---|---|
| `numbers.tex` | `make_numbers.py` | `runs/rq1/rq1_stats.json` |
| `rq4_numbers.tex` | `make_numbers.py` | `runs/rq4/injecagent_validity.json`, `injecagent_frontier_density.json` (pool size 1062 is a constant, see `NOTES_errata.md`) |
| `rq2_numbers.tex`, `rq2_table.tex` (Qwen3-8B reward comparison) | `make_rq2_table.py` | `rq2_paired.py` output for `qwen3-8b-local` |
| `rq2llama_numbers.tex`, `rq2_llama_table.tex` (Llama-3.1-8B, SI) | `make_rq2_table.py` | `rq2_paired.py` output for `llama31-8b-local` |
| `rq2replay_numbers.tex` | `make_rq2_replay_macros.py` | `runs/rq2/rq2_paired_{qwen,llama}.json` |
| `rqtax_numbers.tex` | `rq3_taxonomy.py --models qwen3-8b-yf,qwen3-32b-local,qwen3-8b-local` | `runs/rq1`, `runs/rq2` level-3 rows of the models that pass the tool-call parse gate |
| `rq3_numbers.tex` | `make_rq3_numbers.py` | `testbed/data/frontier.jsonl` |
| `rq2_ops_table.tex`, `ablate_table.tex` | `make_expand_tables.py` | `runs/rq1/census_*.jsonl`, `runs/rq2/*_[0-9].jsonl`, `runs/px/ablate_r3.json` |
| `memo_table.tex` | `make_tables2.py` | `runs/rq2/rq2_paired_*.json` |
| `px_numbers.tex` | `px1a_collision.py` (with and without `--models`), `px2_label.py score`, `make_px_numbers.py` | `runs/rq1`, `runs/rq2`, `testbed/data/frontier.jsonl`, `runs/px/` |
| `gateway_numbers.tex`, `gateway_table.tex` | `gateway_sens.py` | `runs/px/rq3_floor_nosol.json`, `runs/px/gateway_quantiles_release.json` |
| `gate_numbers.tex` (parse-gate shares, SI) | `parse_gate_numbers.py` | `runs/rq1`, `runs/rq1_parsed`, `runs/rq2` |
| `frontier_merged_table.tex`, `rq2_detail_table.tex` | `make_sands_merged.py` | the generated `rq2_suite_table.tex`, `rq2_targets_table.tex`, `rq4_frontier_table.tex` and `paper_static/rq3_permodel_table.tex` |
| `rqtax_table.tex`, `rq4_variants_table.tex` | static layout in `paper_static/`, every number is a macro from the files above | none |
| `figs/frontier_levels.pdf`, `figs/rq4_spread.pdf` | `make_sands_merged.py` | `testbed/data/frontier.jsonl`, `runs/rq4` |
| `figs/rq4_injeccurve.pdf` | `make_figs2.py` | `runs/rq4` |
| `figs/method.pdf` | `method.tex` (TikZ, `pdflatex -output-directory ../out/figs method.tex`) | none |

`reproduce_all.sh` also writes intermediate files that the merged floats read or that the SI of earlier versions used (`rq2_suite_table.tex`, `rq2_targets_table.tex`, `rq4_frontier_table.tex`, `rq1_*` tables, `empty*`, `rq2tmap_numbers.tex`, the other figures). Generated `.tex` files carry a one-line `%` header comment; only that comment can differ from the paper copy.
`make_method_fig.py` draws an earlier matplotlib version of the method figure that the paper no longer uses.

The individual steps are below. All commands run from `analysis/`. Create the output folders first. `out/` is created next to `analysis/`.

```bash
cd analysis
mkdir -p ../out/figs
export AGENTDOJO_SRC=/path/to/agentdojo/src      # skip if agentdojo is pip-installed
export LLM_GATEWAY_KEY=none                        # any value, analysis makes no API call
```

| Script | Command | Produces |
|---|---|---|
| `rq1_stats.py` | `python3 rq1_stats.py ../runs/rq1 --json ../out/rq1_stats.json` | RQ1 per-model level rates and fail-only Spearman rho with Holm, pooled random-effects rho (C2 decision). Output equals the shipped `runs/rq1/rq1_stats.json` |
| `rq1_heldout.py` | `python3 rq1_heldout.py` | held-out split rho macros `out/rq1ho_numbers.tex` |
| `rq2_paired.py` | `python3 rq2_paired.py ../runs/rq2 qwen3-8b-local --json out_q.json` and the same with `llama31-8b-local --json out_l.json` | RQ2 paired A12, Wilcoxon, Holm, memo-hit and fresh-only counts. Equals `runs/rq2/rq2_paired_{qwen,llama}.json` |
| `rq2_tmap.py` | `python3 rq2_tmap.py ../runs/rq2 qwen3-8b-local --json out_t.json` | T-MAP ladder-reward arm comparison. The arm was not rerun after the system-prompt fix and is not in the paper, so no data for it is shipped |
| `make_rq2_table.py` | `python3 make_rq2_table.py out_q.json --macros ../out/rq2_numbers.tex --table ../out/rq2_table.tex --label tab:rq4 --rq RQ4 --no-emptytrace` (Llama: `out_l.json`, `rq2llama_numbers.tex`, `rq2_llama_table.tex`, `tab:rq4llama`) | reward-comparison macros and table, one model per call |
| `rq3_taxonomy.py` | `python3 rq3_taxonomy.py --models qwen3-8b-yf,qwen3-32b-local,qwen3-8b-local` | level-3 anatomy macros `out/rqtax_numbers.tex` on the parse-gate models |
| `px1a_collision.py` | `python3 px1a_collision.py [--models qwen3-8b-yf,qwen3-32b-local,qwen3-8b-local]` | collision rate of the level-3 rule when each failed trace is scored against injection tasks the agent never saw. Equals `runs/px/px1a.json` (with `--models`: `runs/px/px1a_gate.json`) |
| `px2_label.py` | `python3 px2_label.py build`, `label <model>` (needs `LLM_GATEWAY_URL`, `LLM_GATEWAY_KEY`), `score` | two-labeller precision study of level 3. `score` runs offline on the shipped labels and equals `runs/px/px2_summary.json` |
| `make_px_numbers.py` | `python3 make_px_numbers.py` | `out/px_numbers.tex` |
| `make_tables2.py` | `python3 make_tables2.py` | `out/rq1_levelrate_table.tex`, `out/memo_table.tex` |
| `make_tables3.py` | `python3 make_tables3.py` | `out/rq2_suite_table.tex`, `out/rq1_power_table.tex` |
| `make_rq4_tables.py` | `python3 make_rq4_tables.py - ../runs/rq4/injecagent_validity.json > ../out/rq4_tables.tex` | RQ4 InjecAgent validity table, not used by the paper. Equals `runs/rq4/rq4_tables.tex` |
| `make_figs.py` | `python3 make_figs.py` | `out/figs/rq1_levelrate.pdf`, `rq3_density.pdf` |
| `make_figs2.py` | `python3 make_figs2.py` | `out/figs/rq2_box.pdf`, `rq2_fresh.pdf`, `rq4_injeccurve.pdf`, `rq4_frontierbars.pdf` |
| `rq1_samepop.py` | `python3 rq1_samepop.py` | same-population rho for SI Table S3, `out/rq1samepop_table.tex`. Rewrites `runs/px/rq1_samepop.json` |
| `empty_trace.py` | `python3 empty_trace.py` | runs without a tool call, SI Table S4, `out/emptytrace_table.tex` and `out/empty_numbers.tex` |
| `gateway_sens.py` | `python3 gateway_sens.py` | RQ3 and labeller check without gpt-5.6-sol and the SI table of gateway calls with an added prefix, `out/gateway_numbers.tex` and `out/gateway_table.tex`, from `runs/px/rq3_floor_nosol.json` and `runs/px/gateway_quantiles_release.json` |
| `parse_gate_numbers.py`, `s1b_reparse.py` | `python3 parse_gate_numbers.py` | share of error-free runs whose final answer still holds an unexecuted tool call, per model and stage, `out/gate_numbers.tex` |
| `make_sands_merged.py` | `python3 make_sands_merged.py` (after the generators above) | merged main-text floats `frontier_merged_table.tex`, `rq2_detail_table.tex`, `figs/frontier_levels.pdf`, `figs/rq4_spread.pdf` |
| `efmt.py`, `fmtnum.py` | imported | the single number formatter: every rendered number is rounded once, half up, with count ratios passed as exact fractions |
| `prox_v2.py` | imported | proximity ladder v2.1 grader used by every RQ |
| `rq1_cluster.py`, `rq1_demote.py`, `rq1_heldout_undef.py` | `python3 <script>` | RQ1 sensitivity checks (target as cluster, demoted wrong-function runs, undefended held-out units). Write `runs/px/rq1_{cluster,demote,heldout_undef}.json` |
| `rq2_unique.py`, `rq2_crosshost.py` | `python3 <script>` | RQ2 unique-violation view and the one-unit backend sensitivity. Write `runs/rq2/rq2_{unique,crosshost}.json`, read by `make_rq2_replay_macros.py` |
| `rq3_floor_test.py` | `python3 rq3_floor_test.py` | RQ3 density against its collision floor, `runs/px/rq3_floor.json` |
| `px1b_benign.py`, `px3_fresh.py`, `ablate_r3.py` | live runs (need a vLLM serve), then offline summaries | construct-validity checks and the reward-definition ablation. Shipped results: `runs/px/px1b*.json(l)`, `runs/px/px3.json`, `runs/px/ablate_r3.json`, read by `make_px_numbers.py` |
| `make_method_fig2.py` | `python3 make_method_fig2.py` | alternative method figure, not used by the paper |
| `injecagent/injecagent_prox.py` | `INJECAGENT_DIR=/path/to/InjecAgent python3 ../injecagent/injecagent_prox.py` | self-test of the InjecAgent proximity grader used for RQ4 (needs the InjecAgent data) |

The LaTeX table environments are generated as `table`. The paper may wrap them in `table*`.

## Re-running experiments

Needs a GPU server with vLLM. Examples, from `engine/`:

```bash
python3 run_select.py --model qwen3-8b-local --out ../runs/select       # target selection
python3 merge_select.py --model qwen3-8b-local --dir ../runs/select
python3 run_e2.py --model qwen3-8b-local --targets-json ../runs/select/targets_qwen3-8b-local.json \
    --arms random,binary,coverage,prox-v21 --seeds 0,1,2,3,4 --budget 40 --out ../runs/rq2
python3 run_census.py --model qwen3-32b-local --port 8031 --out ../runs/rq1
python3 -m pytest test_engagefuzz.py -q                               # unit tests
cd ../injecagent && python3 injecagent_run.py --model llama31-8b-local --base-url http://127.0.0.1:8023/v1 \
    --n-cases 100 --reps 10 --out ../runs/rq4/injecagent_llama.jsonl
```

`--backend` is a free-form label recorded in each result row (`gpu-a`, `gpu-b`, `gpu-c` in the shipped data).
`run_census.py --parser <name> --gate` records the vLLM tool-call parser per row and stops when the sniff runs fail the tool-call parse gate.
The `qwen3-8b-yf` file label in `runs/rq1/` is the Qwen3-8B census run on the second backend.
Model weights paths in result rows are shown relative (`models/...`).

## Notes on the data

- Result rows are the raw campaign output. Host names and absolute model paths in provenance fields (`backend`, `served_root`) were replaced by neutral labels, nothing else was edited.
- Rows of aborted or superseded runs (Mistral-7B no-tool-call census, Granite sniff run, partial files) are not shipped.
- All Qwen3 rows (RQ1 census of Qwen3-8B and Qwen3-32B, Qwen3-8B campaigns) are the reruns after the system-prompt fix of the harness. The earlier Qwen3 rows were removed from `runs/`; they remain in the history of this repository.
- `runs/rq1_parsed/` holds the Llama-3.1-8B parser smoke run (`llama3_json`) used for the parse-gate disclosure.
- `plan.md` is the dated pre-registration and ruling log, `NOTES_errata.md` lists corrections.
- Synthetic attacker values such as `amy.watson@gmail.com` come from the public benchmarks and test fixtures.
- Model outputs inside traces may contain fake file paths such as `/home/user/...` that the benchmarks' injected tasks mention.

## License

Code and data of this repository: MIT, see `LICENSE`. AgentDojo and InjecAgent keep their own licenses.
