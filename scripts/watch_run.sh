#!/bin/bash
# Uso: bash scripts/watch_run.sh runs/rl_phase1_anchored/run_001
RUN_DIR=${1:-runs/rl_phase1_anchored/run_001}
while true; do
    clear
    echo "=== TRAIN (últimas 5 iters) ==="
    column -ts, "$RUN_DIR/metrics/rl_train.csv" | tail -6
    echo ""
    echo "=== EVAL (últimas 3 filas) ==="
    [ -f "$RUN_DIR/metrics/rl_eval.csv" ] && column -ts, "$RUN_DIR/metrics/rl_eval.csv" | tail -4
    echo ""
    echo "Actualizado: $(date)"
    sleep 300
done
