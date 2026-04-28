#!/bin/bash
set -euo pipefail

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "=== STEP 1: Generación de datos IL ==="
python scripts/tournament/run.py
log "Datos generados."

log "=== STEP 2: Entrenamiento IL ==="
python train.py --config training/il_config.json
log "IL completado."

IL_CKPT=$(ls -t runs/planet_policy_il_v3/*/checkpoints/best.pt | head -1)
log "IL best checkpoint: $IL_CKPT"

log "=== STEP 3: RL Fase 1 ==="
cp training/rl_phase1.json /tmp/rl_phase1_patched.json
sed -i "s|<IL_BEST>|$IL_CKPT|g" /tmp/rl_phase1_patched.json
python train.py --config /tmp/rl_phase1_patched.json
log "Fase 1 completada."

PHASE1_CKPT=$(ls -t runs/rl_phase1_anchored/*/checkpoints/rl_last.pt | head -1)
log "Phase 1 checkpoint: $PHASE1_CKPT"

log "=== STEP 4: RL Fase 2 ==="
cp training/rl_phase2.json /tmp/rl_phase2_patched.json
sed -i "s|rl_phase1_anchored/run_001/checkpoints/rl_last.pt|$PHASE1_CKPT|g" /tmp/rl_phase2_patched.json
python train.py --config /tmp/rl_phase2_patched.json
log "Fase 2 completada."

PHASE2_CKPT=$(ls -t runs/rl_phase2_mixed/*/checkpoints/rl_last.pt | head -1)
log "Phase 2 checkpoint: $PHASE2_CKPT"

log "=== STEP 5: RL Fase 3 ==="
cp training/rl_phase3.json /tmp/rl_phase3_patched.json
sed -i "s|rl_phase2_mixed/run_001/checkpoints/rl_last.pt|$PHASE2_CKPT|g" /tmp/rl_phase3_patched.json
python train.py --config /tmp/rl_phase3_patched.json
log "Fase 3 completada."

log "=== STEP 6: Evaluación final ==="
python train.py eval \
    --checkpoint $(ls -t runs/rl_phase3_league/*/checkpoints/rl_best_winrate.pt | head -1) \
    --opponents heuristic.oracle_sniper \
    --n-matches 30

log "=== PIPELINE COMPLETO ==="
