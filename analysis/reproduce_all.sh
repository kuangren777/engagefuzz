#!/usr/bin/env bash
# Regenerate every paper macro file, table and figure from the frozen data in ../runs.
# Usage: cd analysis && bash reproduce_all.sh [PAPER_DIR]
# With PAPER_DIR (a checkout of the paper sources) each output is compared against the paper copy.
# Needs AGENTDOJO_SRC (or pip-installed agentdojo). pdflatex is only needed for the method figure.
set -euo pipefail
cd "$(dirname "$0")"
export LLM_GATEWAY_KEY="${LLM_GATEWAY_KEY:-none}"
mkdir -p ../out/figs

python3 rq1_stats.py ../runs/rq1 --json ../out/rq1_stats.json > /dev/null
python3 rq2_paired.py ../runs/rq2 qwen3-8b-local --json ../out/rq2_paired_qwen.json > /dev/null
python3 rq2_paired.py ../runs/rq2 llama31-8b-local --json ../out/rq2_paired_llama.json > /dev/null
for f in rq1/rq1_stats rq2/rq2_paired_qwen rq2/rq2_paired_llama; do
  cmp -s "../out/$(basename $f).json" "../runs/$f.json" && echo "same as runs/$f.json" || echo "DIFFERS from runs/$f.json"
done

# same order as the paper build: Qwen3-8B reward comparison in the main text, Llama-3.1-8B in the SI
python3 make_numbers.py
python3 make_rq2_table.py ../out/rq2_paired_qwen.json \
  --macros ../out/rq2_numbers.tex --table ../out/rq2_table.tex --label tab:rq4 --rq RQ4 --no-emptytrace
python3 make_rq2_table.py ../out/rq2_paired_llama.json \
  --macros ../out/rq2llama_numbers.tex --table ../out/rq2_llama_table.tex --label tab:rq4llama --rq RQ4 --no-emptytrace --si
python3 rq1_heldout.py > /dev/null
# level-three anatomy on the models that pass the tool-call parse gate
python3 rq3_taxonomy.py --models qwen3-8b-yf,qwen3-32b-local,qwen3-8b-local > /dev/null
python3 make_rq2_replay_macros.py
python3 make_rq3_numbers.py
python3 make_expand_tables.py
python3 make_tables2.py
python3 make_tables3.py
python3 make_rq2_targets_table.py
python3 make_rq4_frontier_table.py
# construct-validity checks: px1a / px1a_gate / px2_summary are recomputed in place from the shipped
# traces and labels, then compared against the shipped copies
mkdir -p ../out/px_shipped && cp ../runs/px/px1a.json ../runs/px/px1a_gate.json ../runs/px/px2_summary.json ../runs/px/rq1_samepop.json ../out/px_shipped/
python3 px1a_collision.py > /dev/null
python3 px1a_collision.py --models qwen3-8b-yf,qwen3-32b-local,qwen3-8b-local > /dev/null
python3 px2_label.py score > /dev/null
python3 make_px_numbers.py > /dev/null
# SI: same-population rho (rewrites runs/px/rq1_samepop.json), runs without a tool call, gateway prefix table, parse gate
python3 rq1_samepop.py > /dev/null
for f in px1a px1a_gate px2_summary rq1_samepop; do
  cmp -s "../runs/px/$f.json" "../out/px_shipped/$f.json" && echo "same as runs/px/$f.json" || echo "DIFFERS from runs/px/$f.json"
done
python3 empty_trace.py > /dev/null
python3 gateway_sens.py > /dev/null
python3 parse_gate_numbers.py > /dev/null
cp ../paper_static/*.tex ../out/
python3 make_figs.py
python3 make_figs2.py
# merged main-text floats, built from the tables and figures above
python3 make_sands_merged.py
if command -v pdflatex > /dev/null; then
  pdflatex -interaction=nonstopmode -output-directory ../out/figs method.tex > /dev/null
  rm -f ../out/figs/method.aux ../out/figs/method.log
else
  echo "pdflatex not found, skipping out/figs/method.pdf"
fi

if [ $# -ge 1 ]; then
  P="$1"
  for f in ../out/*.tex; do
    b=$(basename "$f")
    [ -f "$P/$b" ] || { echo "not in paper: $b"; continue; }
    cmp -s "$f" "$P/$b" && echo "IDENTICAL $b" || echo "DIFFERS   $b"
  done
  # PDFs embed a creation date and document ID, compare without them
  for f in ../out/figs/*.pdf; do
    b=$(basename "$f")
    python3 - "$f" "$P/figs/$b" "$b" <<'EOF'
import re, sys
st = lambda p: re.sub(rb"/(CreationDate|ModDate) \(D:[^)]*\)|/ID ?\[[^\]]*\]", b"", open(p, "rb").read())
print(("IDENTICAL (except date/ID) " if st(sys.argv[1]) == st(sys.argv[2]) else "DIFFERS   ") + "figs/" + sys.argv[3])
EOF
  done
fi
