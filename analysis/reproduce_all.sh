#!/usr/bin/env bash
# Regenerate every paper macro file, table and figure from the frozen data in ../runs.
# Usage: cd analysis && bash reproduce_all.sh [PAPER_DIR]
# With PAPER_DIR (a checkout of the paper sources) each output is compared against the paper copy.
# Needs AGENTDOJO_SRC (or pip-installed agentdojo). pdflatex is only needed for the method figure.
set -euo pipefail
cd "$(dirname "$0")"
export CAMEL_HUB_KEY="${CAMEL_HUB_KEY:-none}"
mkdir -p ../out/figs

python3 rq1_stats.py ../runs/rq1 --json ../out/rq1_stats.json > /dev/null
python3 rq2_paired.py ../runs/rq2 qwen3-8b-local --json ../out/rq2_paired_qwen.json > /dev/null
python3 rq2_paired.py ../runs/rq2 llama31-8b-local --json ../out/rq2_paired_llama.json > /dev/null
python3 rq2_tmap.py ../runs/rq2 qwen3-8b-local --json ../out/rq2_tmap_qwen.json > /dev/null
for f in rq1/rq1_stats rq2/rq2_paired_qwen rq2/rq2_paired_llama rq2/rq2_tmap_qwen; do
  cmp -s "../out/$(basename $f).json" "../runs/$f.json" && echo "same as runs/$f.json" || echo "DIFFERS from runs/$f.json"
done

python3 make_numbers.py
python3 make_rq2_table.py ../out/rq2_paired_qwen.json ../out/rq2_paired_llama.json \
  --macros ../out/rq2_numbers.tex --table ../out/rq2_table.tex
python3 rq1_heldout.py > /dev/null
python3 rq3_taxonomy.py > /dev/null
python3 make_rq2_replay_macros.py
python3 make_rq3_numbers.py
python3 make_expand_tables.py
python3 make_tables2.py
python3 make_tables3.py
python3 make_rq2_targets_table.py
python3 make_rq4_frontier_table.py
cp ../paper_static/*.tex ../out/
python3 make_figs.py
python3 make_figs2.py
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
