#!/usr/bin/env bash
# Overnight Stage 2 run, trimmed to fit (~7 min per full PySR search on a busy CPU). Cached; rerun resumes.
cd /c/Users/abhis/SFIC-GP-for-Economic-Shock-Prediction || exit 1
export PYTHONIOENCODING=utf-8
PY=".venv/Scripts/python -W ignore -m src.v1.run"
JJ=${JJ:-5}
step() { echo "=== $(date +%H:%M) $*"; }

step "main: pysr_mdl (5 seeds)"
$PY --models pysr_mdl --seeds 5 --julia-jobs $JJ || echo "FAILED main pysr_mdl"
step "main: pysr_anchor + pysr_mv, pysr_bag (2 seeds x 8 bags)"
$PY --models pysr_anchor,pysr_mv,pysr_bag --seeds 2 --julia-jobs $JJ || echo "FAILED main bagged"
step "main: pysr_jk (1 seed)"
$PY --models pysr_jk --seeds 1 --julia-jobs $JJ || echo "FAILED main pysr_jk"
step "main: pysr_enet (3 seeds)"
$PY --models pysr_enet --seeds 3 --julia-jobs $JJ || echo "FAILED main pysr_enet"
step "report (interim)"
.venv/Scripts/python -W ignore -m src.v1.report > results/v1/report_stdout.txt 2>&1 || echo "FAILED report interim"

for k in 0 1 2 3 4; do
  step "placebo$k"
  $PY --variant placebo$k --models linear_beta,linear_fomc,linear_char,linear_jk --seeds 1 --jobs 4 || echo "FAILED placebo$k linear"
  $PY --variant placebo$k --models pysr_anchor,pysr_mv,pysr_mdl --seeds 1 --julia-jobs $JJ || echo "FAILED placebo$k pysr"
done

step "main: gp_sens (3 seeds)"
$PY --models gp_sens --seeds 3 --jobs 8 || echo "FAILED main gp_sens"
step "report"
.venv/Scripts/python -W ignore -m src.v1.report > results/v1/report_stdout.txt 2>&1 || echo "FAILED report"
step "ALL DONE"
