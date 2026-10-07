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
| `analysis/` | Statistics, tables, figures and LaTeX macros |
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
export CAMEL_HUB_URL=https://your-gateway.example.com   # without the trailing /v1
export CAMEL_HUB_KEY=...                                # your key
export EF_JUDGE=gpt-5.5                                 # judge model name on your gateway (optional)
```

They can also be placed in `~/.config/engagefuzz/env` as `KEY=value` lines. Runs that use only local vLLM models never call the gateway.
Analysis from the frozen data needs neither the gateway nor a GPU, only `AGENTDOJO_SRC` (the analysis imports AgentDojo task suites).

## Reproduce the paper numbers from `runs/`

All commands run from `analysis/`. Create the output folders first. `out/` is created next to `analysis/`.

```bash
cd analysis
mkdir -p ../out/figs
export AGENTDOJO_SRC=/path/to/agentdojo/src      # skip if agentdojo is pip-installed
export CAMEL_HUB_KEY=none                        # any value, analysis makes no API call
```

| Script | Command | Produces |
|---|---|---|
| `rq1_stats.py` | `python3 rq1_stats.py ../runs/rq1 --json ../runs/rq1/rq1_stats.json` | RQ1 per-model level rates and fail-only Spearman rho with Holm, pooled random-effects rho (C2 decision). Output equals the shipped `runs/rq1/rq1_stats.json` |
| `rq1_heldout.py` | `python3 rq1_heldout.py` | held-out split rho macros `out/rq1ho_numbers.tex` |
| `rq2_paired.py` | `python3 rq2_paired.py ../runs/rq2 qwen3-8b-local --json out_q.json` and the same with `llama31-8b-local --json out_l.json` | RQ2 paired A12, Wilcoxon, Holm, memo-hit and fresh-only counts. Equals `runs/rq2/rq2_paired_{qwen,llama}.json` |
| `rq2_tmap.py` | `python3 rq2_tmap.py ../runs/rq2 qwen3-8b-local --json out_t.json` | T-MAP ladder-reward arm comparison. Equals `runs/rq2/rq2_tmap_qwen.json` |
| `make_rq2_table.py` | `python3 make_rq2_table.py out_q.json out_l.json --macros ../out/rq2_numbers.tex --table ../out/rq2_table.tex` | RQ2 macros and main RQ2 table |
| `rq3_taxonomy.py` | `python3 rq3_taxonomy.py` | level-3 failure taxonomy macros `out/rqtax_numbers.tex` |
| `make_tables2.py` | `python3 make_tables2.py` | `out/rq1_levelrate_table.tex`, `out/memo_table.tex` |
| `make_tables3.py` | `python3 make_tables3.py` | `out/rq2_suite_table.tex`, `out/rq1_power_table.tex` |
| `make_rq4_tables.py` | `python3 make_rq4_tables.py - ../runs/rq4/injecagent_validity.json > ../out/rq4_tables.tex` | RQ4 InjecAgent validity table. Equals `runs/rq4/rq4_tables.tex` |
| `make_figs.py` | `python3 make_figs.py` | `out/figs/rq1_levelrate.pdf`, `rq3_density.pdf` |
| `make_figs2.py` | `python3 make_figs2.py` | `out/figs/rq2_box.pdf`, `rq2_fresh.pdf`, `rq4_injeccurve.pdf`, `rq4_frontierbars.pdf` |
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

`--backend` is a free-form label recorded in each result row (`gpu-a`, `gpu-b` in the shipped data).
The `qwen3-8b-yf` file label in `runs/rq1/` is the Qwen3-8B census run on the second backend.
Model weights paths in result rows are shown relative (`models/...`).

## Notes on the data

- Result rows are the raw campaign output. Host names and absolute model paths in provenance fields (`backend`, `served_root`) were replaced by neutral labels, nothing else was edited.
- Rows of aborted or superseded runs (Mistral-7B no-tool-call census, Granite sniff run, partial files) are not shipped.
- `plan.md` is the dated pre-registration and ruling log, `NOTES_errata.md` lists corrections.
- Synthetic attacker values such as `amy.watson@gmail.com` come from the public benchmarks and test fixtures.
- Model outputs inside traces may contain fake file paths such as `/home/user/...` that the benchmarks' injected tasks mention.

## License

Code and data of this repository: MIT, see `LICENSE`. AgentDojo and InjecAgent keep their own licenses.
