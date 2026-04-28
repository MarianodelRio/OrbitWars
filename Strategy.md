# OrbitWars — Training Strategy

**Objetivo:** pasar de cero datos a un bot neural competitivo en Kaggle. Este documento se lee una vez y se ejecuta en orden. Cada sección tiene comandos exactos.

---

## 0. Fixes obligatorios antes de entrenar (< 20 min)

Hay tres bugs que bloquean un pipeline limpio. Son cambios pequeños en archivos conocidos. Hay que resolverlos **antes** de lanzar cualquier entrenamiento.

### Bug 1 — `train.py` ignora `lstm_bypass`, `n_layers`, `ffn_hidden`

`train.py` construye `PlanetPolicyConfig(...)` manualmente pero no pasa estos tres campos, que el dataclass sí expone. El resultado: el JSON de config los declara pero el modelo siempre arranca con los defaults (`n_layers=4, ffn_hidden=768, lstm_bypass=False`).

Riesgo real: `il_config.json` tiene `"lstm_bypass": true` — **ignorado**. El bot IL entrena con LSTM aunque el config diga lo contrario. Si luego el RL usa `lstm_bypass=true`, el warmstart falla porque los pesos del LSTM no existen.

**Decisión: usar LSTM en ambos (IL y RL). Poner `lstm_bypass: false` explícitamente en ambos configs y añadir las tres líneas al `train.py`.**

Archivo: [train.py](train.py), líneas 75–87 (RL) y 139–151 (IL). En ambos bloques, dentro de `PlanetPolicyConfig(...)`, añadir:
```python
n_layers=model_config_dict.get("n_layers", 4),
ffn_hidden=model_config_dict.get("ffn_hidden", 768),
lstm_bypass=model_config_dict.get("lstm_bypass", False),
```

### Bug 2 — `PotentialReward` no incluye naves en flota

[training/rewards/potential.py:65](training/rewards/potential.py#L65) calcula `my_ships` solo desde planetas. Pero el scoring de Kaggle cuenta **planetas + flotas**. El efecto: el agente aprende a no enviar flotas porque cada flota lanzada reduce momentáneamente su "potencial" y genera reward negativo.

Archivo: [training/rewards/potential.py:54-72](training/rewards/potential.py#L54). En el método `_potential`, añadir las naves en tránsito:
```python
def _potential(self, obs: dict, player: int) -> float:
    planets = obs.get("planets", [])
    fleets  = obs.get("fleets", [])
    if not planets:
        return 0.0
    my_planets     = sum(1 for p in planets if p[1] == player)
    total_planets  = max(len(planets), 1)
    my_production  = sum(p[6] for p in planets if p[1] == player)
    total_prod     = max(sum(p[6] for p in planets), 1)
    my_ships       = sum(p[5] for p in planets if p[1] == player) \
                   + sum(f[6] for f in fleets  if f[1] == player)
    log_ships_share = math.log(1 + my_ships) / math.log(1001)
    return (
        self.w_production * (my_production / total_prod)
        + self.w_planets  * (my_planets   / total_planets)
        + self.w_ships    * log_ships_share
    )
```

### Bug 3 — `rl_best_winrate.pt` nunca se escribe

[training/trainers/rl_trainer.py:403-418](training/trainers/rl_trainer.py#L403) llama a `save_rl_checkpoint(...)` pero siempre con `is_best_winrate=False` (default). El archivo `rl_best_winrate.pt` — el mejor checkpoint para submisión — nunca existe.

En `rl_trainer.py`, añadir un tracker de mejor win rate y pasarlo:
```python
# en __init__ o _setup:
self._best_mean_winrate = 0.0

# dentro del bloque eval_every (tras evaluator.run):
mean_wr = sum(r.get("win_rate", 0.0) for r in eval_results.values()) / max(len(eval_results), 1)
is_best = mean_wr > self._best_mean_winrate
if is_best:
    self._best_mean_winrate = mean_wr
self._ckpt_manager.save_rl_checkpoint(..., is_best_winrate=is_best)
```

### Bug 4 — `Makefile` apunta a los scripts viejos

`make train` ejecuta `scripts/train_il.py` y `make train-rl` ejecuta `scripts/train_rl.py`. El punto de entrada unificado es ahora `train.py`. Actualizar el Makefile:
```makefile
train:
	$(PYTHON) train.py --config training/il_config.json

train-rl:
	$(PYTHON) train.py --config $(CONFIG)
```

### Verificar tras los fixes
```bash
make test-quick
make train --dry-run
make train-rl CONFIG=training/rl_config.json --dry-run
```

---

## 1. Setup del entorno GPU (GCP)

### VM recomendada

| Campo | Valor |
|---|---|
| Machine type | `n1-standard-8` (8 vCPU, 30 GB RAM) |
| GPU | `nvidia-tesla-t4` (mínimo). L4 si disponible — mismo precio, 2× throughput bf16 |
| Disco | 200 GB balanced persistent |
| Imagen | Deep Learning VM (`pytorch-latest-gpu`) — PyTorch 2.x + CUDA 12.1 preinstalado |
| Preemptible | Solo para generación de datos (es reanudable). NO para RL (pierde el buffer) |

El cuello de botella de RL **es la CPU** (`kaggle_environments` es single-threaded por episode), no la GPU. Los 8 vCPUs importan más que el tier de GPU para el throughput total.

### Provisionar
```bash
gcloud compute instances create orbit-train \
    --zone=us-central1-a \
    --machine-type=n1-standard-8 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=200GB --boot-disk-type=pd-balanced \
    --metadata="install-nvidia-driver=True" \
    --maintenance-policy=TERMINATE
```

### Transferir código
```bash
# Desde Windows (PowerShell o git bash) — sube solo el código, sin data/ ni runs/:
gcloud compute scp --recurse --compress --zone=us-central1-a \
    C:/Users/mariano.del.rio/OrbitWars orbit-train:~/OrbitWars \
    --exclude="data/*" --exclude="runs/*" --exclude=".venv/*"

# O usar GitHub (más limpio para iteraciones):
# git push en local → git clone en la VM
```

### Install en la VM
```bash
cd ~/OrbitWars
bash setup.sh
source .venv/bin/activate
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
make test-quick
make train --dry-run
make train-rl CONFIG=training/rl_config.json --dry-run
```

Los tres primeros comandos confirman: CUDA accesible, tests verdes, configs parseables. Si alguno falla, parar aquí.

---

## 2. Generación de datos para IL

### Cuántos episodios necesitamos

**Target: 1500 episodios.** Razonamiento:
- Cada episodio ≈ 500 turnos × 2 perspectivas = 1000 muestras
- 1500 × 1000 = **1.5M muestras** para IL
- Con batch=256 y 50 epochs ≈ 290k pasos de gradiente — zona cómoda para 7M params
- 500 episodios es el mínimo viable (bot funcional pero plateau temprano)
- Más de 2000 es sobreingeniería salvo que amplíes la diversidad de bots

### Qué bots usar

| Par | Episodios | Por qué |
|---|---|---|
| `oracle_sniper` vs `oracle_sniper` | 600 | El mejor heurístico genera las mejores acciones; self-play garantiza calidad en ambas perspectivas |
| `oracle_sniper` vs `sniper` | 400 | Presión asimétrica — oracle gana ~70%, diversifica tácticamente |
| `oracle_sniper` vs `scoring` | 300 | Estilo diferente (más conservador), diversifica el prior |
| `sniper` vs `scoring` | 200 | Par más débil; sin esto el modelo sobre-aprende micropatrones de oracle |

**NO incluir `baseline`** en los datos de IL — sus acciones son ruido puro, contamina el prior. Úsalo solo para evaluación.

**No filtrar `winner_only`:** el head de valor aprende el outcome (win/loss/draw) de la etiqueta `value_target`. Las acciones del perdedor en partidas entre buenos heurísticos siguen siendo cercanas al óptimo — perdieron por pequeñas asimetrías de mapa, no por jugar mal.

### Comando para generarlo todo de una vez

El runner de torneo hace round-robin con self-play. Editar [scripts/tournament/config.json](scripts/tournament/config.json):

```json
{
  "n_matches": 250,
  "steps": 500,
  "save_log": true,
  "self_play": true,
  "save_data": true,
  "bots": {
    "oracle_sniper": "bots.heuristic.oracle_sniper:agent_fn",
    "sniper":        "bots.heuristic.sniper:agent_fn",
    "scoring":       "bots.scoring.bot:agent_fn"
  }
}
```

Con `n_matches=250` y `self_play=true`:
- Pares cruzados: C(3,2)=3 pares × 250 = **750 episodios**
- Self-play: 3 bots × 250 = **750 episodios**
- **Total: 1500 episodios** con un solo comando

```bash
make data
```

Los `.h5` se escriben en `data/tournaments/<timestamp>/`. `DataCatalog` los descubre automáticamente.

### Tiempo estimado

- ~2-3 s/episodio en `n1-standard-8`
- 1500 episodios ≈ **45-75 minutos en serie**
- Para acelerar: lanzar 4 procesos en paralelo (cada uno apuntando a diferente config o con diferente n_matches parcial) — 4× vCPUs → ~15 minutos

### Verificación post-generación
```bash
python -c "
from dataset.catalog import DataCatalog
c = DataCatalog.scan()
print(f'Episodios: {len(c.episodes)}')
print(f'Draws: {sum(1 for e in c.episodes if e.winner == -1)}')
import statistics
steps = [e.total_steps for e in c.episodes]
print(f'Steps mediana: {statistics.median(steps):.0f}, min: {min(steps)}')
"
```
Esperado: ~1500 episodios, < 3% draws, mediana de pasos 480-500.

---

## 3. Entrenamiento IL

### Por qué mantener el LSTM (no bypass)

El juego tiene información parcialmente observable que el transformador no puede reconstruir por sí solo:
- Comets spawnan exactamente en turnos 50/150/250/350/450 — necesitas recordar cuántos has capturado ya
- Flotas que lanzaste hace 3 turnos están en vuelo — sin memoria, podrías lanzar otra flota redundante al mismo destino

El LSTM tiene ~150k params sobre 7M totales: coste despreciable, beneficio real. Con el fix del Bug 1, ambos configs (IL y RL) deben tener `"lstm_bypass": false`.

### Config IL recomendada

El [training/il_config.json](training/il_config.json) actual es bueno. Cambios para aplicar:

```json
{
  "lstm_bypass": false,
  "eval_opponents": ["heuristic.baseline", "heuristic.sniper", "heuristic.oracle_sniper"],
  "eval_every": 5,
  "n_eval_matches": 20
}
```

El resto del config ya está bien tuneado:
- `epochs: 50` con `early_stopping_patience: 3` — se detiene en ~épocas 20-30 típicamente
- `lr: 3e-4` + cosine_with_warmup — schedule correcto para transformers
- `batch_size: 256` + `use_amp: true` con `bfloat16` — máximo throughput en T4
- `augment_reflection: true` — refleja el mapa horizontalmente/verticalmente, dobla el dataset efectivo
- `action_type_loss_weight: 2.0` — la mayoría de planetas hacen NO_OP cada turno; el peso + `use_class_weights=true` compensa el desequilibrio

### Lanzamiento
```bash
make train-bg
```

La primera época construye el cache HDF5 en `data/cache/il_planet_policy.h5` (~10 min para 1500 episodios). Las siguientes épocas son ~3-5 min/época en T4.

### Criterio de salida IL

El bot IL está listo para RL cuando:
1. `early_stopping` ha parado el entrenamiento (val_loss plató ≥ 3 evals)
2. Win rate vs `heuristic.baseline` ≥ **0.85**
3. Win rate vs `heuristic.sniper` ≥ **0.40**

Verificar en `runs/planet_policy_il_v3/run_001/metrics/`:
```bash
# Últimas evaluaciones IL
cat runs/planet_policy_il_v3/run_001/metrics/val_metrics.csv | tail -5
ls runs/planet_policy_il_v3/run_001/eval/
```

Si después de 50 épocas no alcanzas el 0.40 vs sniper, **el problema es el dataset**: añade 500 episodios más de `oracle_sniper` self-play.

El archivo de salida que usarás para RL es:
```
runs/planet_policy_il_v3/run_001/checkpoints/best.pt
```

---

## 4. Entrenamiento RL — tres fases

Antes de arrancar, decide el path del checkpoint IL (output de la sección anterior) y reemplaza `<IL_BEST>` en los configs de abajo con la ruta real.

### Fase 1 — Ancla al prior IL con heurísticos (300 iteraciones, ~5h en T4)

Objetivo: no colapsar el prior IL. Aprender a jugar el juego contra oponentes variados sin olvidar cómo hacerlo.

Crear [training/rl_phase1.json](training/rl_phase1.json):
```json
{
  "run_name": "rl_phase1_anchored",
  "run_id": "",
  "device": "auto",
  "seed": 42,
  "total_iterations": 300,

  "model_config": {
    "Dp": 24, "Df": 16, "Dg": 16,
    "E": 192, "F": 128, "G": 384,
    "max_planets": 50, "max_fleets": 200,
    "n_amount_bins": 8, "n_heads": 8,
    "n_layers": 4, "ffn_hidden": 768,
    "dropout": 0.1, "lstm_bypass": false
  },

  "n_rollout_steps": 2048,
  "steps_per_episode": 500,
  "ppo_epochs": 4,
  "ppo_batch_size": 256,
  "clip_eps": 0.2,
  "vf_coef": 0.5,
  "max_grad_norm": 0.5,
  "lr": 0.0003,
  "lr_schedule": "cosine",
  "normalize_advantages": true,

  "gamma": 0.997,
  "gae_lambda": 0.95,

  "w_planets": 0.5,
  "w_production": 1.0,
  "w_ships": 0.1,
  "reward_lambda": 0.5,
  "r_terminal_win": 10.0,
  "r_terminal_loss": -10.0,
  "r_terminal_margin_coef": 5.0,
  "r_event_capture_enemy": 0.5,
  "r_event_capture_comet": 0.2,
  "r_event_eliminate_opponent": 1.0,
  "r_event_lose_planet": -0.3,
  "r_explore": 0.01,
  "explore_iterations": 200,

  "entropy_coef_action_type": 0.02,
  "entropy_coef_target": 0.005,
  "entropy_coef_amount": 0.005,

  "self_play_prob": 0.0,
  "heuristic_opponents": [
    "bots.heuristic.baseline:agent_fn",
    "bots.heuristic.sniper:agent_fn",
    "bots.scoring.bot:agent_fn",
    "bots.heuristic.oracle_sniper:agent_fn"
  ],
  "frozen_checkpoint": "<IL_BEST>",
  "max_snapshots": 5,
  "snapshot_every": 50,

  "bc_policy_path": "<IL_BEST>",
  "kl_bc_coef_start": 1.0,
  "kl_bc_coef_end": 0.1,
  "kl_bc_coef_decay_iters": 200,

  "il_data_cache_path": "data/cache/il_planet_policy.h5",
  "il_distill_ratio": 0.10,

  "eval_every": 25,
  "n_eval_matches": 20,
  "eval_opponents": ["heuristic.baseline", "heuristic.sniper", "scoring.bot", "heuristic.oracle_sniper"],
  "save_every": 50
}
```

**Parámetros clave de esta fase:**
- `frozen_checkpoint` inicializa los pesos del modelo desde el checkpoint IL
- `bc_policy_path` carga una copia congelada separada del mismo checkpoint IL para el término KL. Son dos instancias distintas: una se entrena, la otra es fija.
- `il_distill_ratio: 0.10` — 10% de los minibatches PPO se reemplazan por loss CE sobre datos IL. Previene el olvido catastrófico.
- `self_play_prob: 0.0` — sin self-play en fase 1; el modelo es demasiado débil para entrenar contra sí mismo
- La cache IL (`il_data_cache_path`) debe existir antes de arrancar (la crea el IL trainer automáticamente)

**Lanzamiento:**
```bash
make train-phase1
```

**Criterio de salida:** Win rate vs `heuristic.sniper` ≥ 0.65 Y vs `heuristic.oracle_sniper` ≥ 0.30 en las últimas 2 evaluaciones. Si no lo alcanza en 300 iters, extender 100 más (el trainer auto-resume con el mismo config).

---

### Fase 2 — Self-play mixto (500 iteraciones, ~9h en T4)

Objetivo: construir un pool de oponentes históricos propio mientras se mantiene ancla a los heurísticos.

Crear [training/rl_phase2.json](training/rl_phase2.json):
```json
{
  "run_name": "rl_phase2_mixed",
  "run_id": "",
  "device": "auto",
  "seed": 42,
  "total_iterations": 500,

  "model_config": {
    "Dp": 24, "Df": 16, "Dg": 16,
    "E": 192, "F": 128, "G": 384,
    "max_planets": 50, "max_fleets": 200,
    "n_amount_bins": 8, "n_heads": 8,
    "n_layers": 4, "ffn_hidden": 768,
    "dropout": 0.1, "lstm_bypass": false
  },

  "n_rollout_steps": 2048,
  "steps_per_episode": 500,
  "ppo_epochs": 4,
  "ppo_batch_size": 256,
  "clip_eps": 0.2,
  "vf_coef": 0.5,
  "max_grad_norm": 0.5,
  "lr": 0.0003,
  "lr_schedule": "cosine",
  "normalize_advantages": true,

  "gamma": 0.997,
  "gae_lambda": 0.95,

  "w_planets": 0.5,
  "w_production": 1.0,
  "w_ships": 0.1,
  "reward_lambda": 0.5,
  "r_terminal_win": 10.0,
  "r_terminal_loss": -10.0,
  "r_terminal_margin_coef": 5.0,
  "r_event_capture_enemy": 0.5,
  "r_event_capture_comet": 0.2,
  "r_event_eliminate_opponent": 1.0,
  "r_event_lose_planet": -0.3,
  "r_explore": 0.0,
  "explore_iterations": 0,

  "entropy_coef_action_type": 0.02,
  "entropy_coef_target": 0.005,
  "entropy_coef_amount": 0.005,

  "self_play_prob": 0.4,
  "heuristic_opponents": [
    "bots.heuristic.sniper:agent_fn",
    "bots.scoring.bot:agent_fn",
    "bots.heuristic.oracle_sniper:agent_fn"
  ],
  "frozen_checkpoint": "runs/rl_phase1_anchored/run_001/checkpoints/rl_last.pt",
  "max_snapshots": 5,
  "snapshot_every": 50,

  "bc_policy_path": "",
  "il_data_cache_path": "",
  "il_distill_ratio": 0.0,

  "eval_every": 50,
  "n_eval_matches": 20,
  "eval_opponents": ["heuristic.baseline", "heuristic.sniper", "scoring.bot", "heuristic.oracle_sniper"],
  "save_every": 50
}
```

**Cambios clave vs fase 1:**
- Se elimina el ancla KL-to-BC (`bc_policy_path: ""`) — ya aprendió a jugar, no necesita la muleta
- Se elimina IL distillation (`il_distill_ratio: 0.0`) — igual
- `self_play_prob: 0.4` — 40% self-play. Los snapshots se acumulan cada 50 iters (pool de 5)
- Se elimina `baseline` de los oponentes — es demasiado fácil y sesga la entropía al alza
- `r_explore: 0.0` — sin bonus de exploración ya (el bot ya sabe moverse)
- `frozen_checkpoint` carga los pesos del último checkpoint de fase 1

**Lanzamiento:**
```bash
make train-phase2
```

**Criterio de salida:** Win rate vs `heuristic.oracle_sniper` ≥ 0.55 de forma consistente en las últimas 3 evaluaciones.

---

### Fase 3 — Liga de self-play (700 iteraciones, ~12h en T4)

Objetivo: extraer los últimos puntos de habilidad con un pool de oponentes más profundo y LR más bajo.

Crear [training/rl_phase3.json](training/rl_phase3.json):
```json
{
  "run_name": "rl_phase3_league",
  "run_id": "",
  "device": "auto",
  "seed": 42,
  "total_iterations": 700,

  "model_config": {
    "Dp": 24, "Df": 16, "Dg": 16,
    "E": 192, "F": 128, "G": 384,
    "max_planets": 50, "max_fleets": 200,
    "n_amount_bins": 8, "n_heads": 8,
    "n_layers": 4, "ffn_hidden": 768,
    "dropout": 0.1, "lstm_bypass": false
  },

  "n_rollout_steps": 2048,
  "steps_per_episode": 500,
  "ppo_epochs": 4,
  "ppo_batch_size": 256,
  "clip_eps": 0.2,
  "vf_coef": 0.5,
  "max_grad_norm": 0.5,
  "lr": 0.0001,
  "lr_schedule": "cosine",
  "normalize_advantages": true,

  "gamma": 0.997,
  "gae_lambda": 0.95,

  "w_planets": 0.5,
  "w_production": 1.0,
  "w_ships": 0.1,
  "reward_lambda": 0.5,
  "r_terminal_win": 10.0,
  "r_terminal_loss": -10.0,
  "r_terminal_margin_coef": 5.0,
  "r_event_capture_enemy": 0.5,
  "r_event_capture_comet": 0.2,
  "r_event_eliminate_opponent": 1.0,
  "r_event_lose_planet": -0.3,
  "r_explore": 0.0,
  "explore_iterations": 0,

  "entropy_coef_action_type": 0.02,
  "entropy_coef_target": 0.005,
  "entropy_coef_amount": 0.005,

  "self_play_prob": 0.7,
  "heuristic_opponents": [
    "bots.heuristic.oracle_sniper:agent_fn"
  ],
  "frozen_checkpoint": "runs/rl_phase2_mixed/run_001/checkpoints/rl_last.pt",
  "max_snapshots": 8,
  "snapshot_every": 30,

  "bc_policy_path": "",
  "il_data_cache_path": "",
  "il_distill_ratio": 0.0,

  "eval_every": 50,
  "n_eval_matches": 20,
  "eval_opponents": ["heuristic.baseline", "heuristic.sniper", "heuristic.oracle_sniper"],
  "save_every": 50
}
```

**Cambios clave vs fase 2:**
- `lr: 0.0001` — LR 3× más bajo para refinamiento fino
- `self_play_prob: 0.7` — mayoría self-play
- `max_snapshots: 8` + `snapshot_every: 30` — liga más profunda y frecuente
- Solo `oracle_sniper` como heurístico de ancla — es el único que sigue siendo un oponente real

**Lanzamiento:**
```bash
make train-phase3
```

**Criterio de salida:** Win rate vs `heuristic.oracle_sniper` ≥ 0.70. A este nivel tienes un bot Kaggle-ready.

---

## 5. Monitorización (mientras el training corre)

### Métricas clave y sus rangos saludables

| Columna (CSV) | Esperado | Señal de alarma |
|---|---|---|
| `mean_ep_reward` | subiendo; entre +5 y +15 en iter 200 | plano u oscilando cerca de 0 |
| `policy_loss` | pequeño (\|x\| < 0.05) | saltos > 0.5 |
| `value_loss` | bajando hasta plateau | creciendo después del iter 100 |
| `entropy` | 1.5 → 0.6 a lo largo del training | < 0.3 (colapso) o sin cambio (sin aprendizaje) |
| `approx_kl` | 0.005 – 0.03 | > 0.05 (clip demasiado suelto) o < 0.001 (no aprende) |
| `clip_fraction` | 0.05 – 0.20 | > 0.30 (LR demasiado alto) |
| `explained_variance` | subiendo hacia 0.7 | negativo (cabeza de valor rota) |

Los archivos de métricas están en `runs/<nombre>/<run_id>/metrics/`.

### Comandos shell (sin Python)

```bash
# Ver las últimas 5 iteraciones de training en tabla
column -ts, runs/rl_phase1_anchored/run_001/metrics/rl_train.csv | tail -6

# Seguir en vivo
tail -f runs/rl_phase1_anchored/run_001/metrics/rl_train.csv

# Ver resultados de eval para un oponente
grep oracle_sniper runs/rl_phase1_anchored/run_001/metrics/rl_eval.csv | column -ts,

# Resumen rápido: iteración, entropía, reward medio
awk -F, 'NR>1{print $1, $8, $9}' runs/rl_phase1_anchored/run_001/metrics/rl_train.csv | tail -20

# Ver log de training
tail -50 runs/rl_phase1.log
```

### Script de monitorización rápida

Guardar como [scripts/watch_run.sh](scripts/watch_run.sh):
```bash
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
```

```bash
make watch RUN=runs/rl_phase1_anchored/run_001
```

---

## 6. Gestión de bots

### Estructura de archivos

```
runs/
  planet_policy_il_v3/run_001/checkpoints/
    best.pt            ← mejor validación IL → usar como punto de partida RL
    last.pt            ← última época IL
    epoch_NNN.pt       ← epoch por epoch

  rl_phase1_anchored/run_001/checkpoints/
    rl_last.pt         ← último checkpoint RL (auto-resume)
    rl_iter_000300.pt  ← checkpoint explícito en iter 300
    rl_best_winrate.pt ← mejor win rate medio (requiere Bug 3 fix)
    snapshots/
      snap_000050.pt   ← snapshots del pool de oponentes

  rl_phase2_mixed/run_001/checkpoints/...
  rl_phase3_league/run_001/checkpoints/...
```

### Comparar dos checkpoints rápidamente

Editar [scripts/matches/config.json](scripts/matches/config.json):
```json
{
  "mode": "evaluate",
  "bot1": "bots.neural.bot:agent_fn?checkpoint=runs/planet_policy_il_v3/run_001/checkpoints/best.pt",
  "bot2": "bots.neural.bot:agent_fn?checkpoint=runs/rl_phase1_anchored/run_001/checkpoints/rl_last.pt",
  "n_matches": 50,
  "steps": 500,
  "save_log": true,
  "save_data": false
}
```
```bash
make match
```

### Torneo de tres vías (IL vs RL vs oracle_sniper)

Editar [scripts/tournament/config.json](scripts/tournament/config.json):
```json
{
  "n_matches": 50,
  "steps": 500,
  "save_log": true,
  "self_play": false,
  "save_data": false,
  "bots": {
    "il_best":      "bots.neural.bot:agent_fn?checkpoint=runs/planet_policy_il_v3/run_001/checkpoints/best.pt",
    "rl_phase1":    "bots.neural.bot:agent_fn?checkpoint=runs/rl_phase1_anchored/run_001/checkpoints/rl_last.pt",
    "rl_phase3":    "bots.neural.bot:agent_fn?checkpoint=runs/rl_phase3_league/run_001/checkpoints/rl_best_winrate.pt",
    "oracle":       "bots.heuristic.oracle_sniper:agent_fn"
  }
}
```
```bash
make tournament
```
Imprime leaderboard ELO. Si `rl_phase3` supera a `oracle` en ELO por ≥ 50 puntos y gana ≥ 65% contra él, el bot está listo para Kaggle.

### Cargar un bot para inferencia manual
```python
from bots.neural.bot import NeuralBot
bot = NeuralBot.load("runs/rl_phase3_league/run_001/checkpoints/rl_best_winrate.pt")
actions = bot.act(obs)
bot.reset()  # entre episodios
```

### Evaluar un checkpoint desde CLI
```bash
make eval CKPT=runs/rl_phase3_league/run_001/checkpoints/rl_best_winrate.pt
```

### Promover a submisión Kaggle
```bash
# Copiar el mejor checkpoint
cp runs/rl_phase3_league/run_001/checkpoints/rl_best_winrate.pt submission/checkpoint.pt

# Empaquetar y subir (requiere KAGGLE_API_TOKEN)
make submit-neural
```

---

## 7. Pipeline completo automatizado

Guardar como [scripts/run_pipeline.sh](scripts/run_pipeline.sh). Ejecutar una vez y irse:

```bash
#!/bin/bash
set -euo pipefail

IL_CKPT="runs/planet_policy_il_v3/run_001/checkpoints/best.pt"
PHASE1_CKPT="runs/rl_phase1_anchored/run_001/checkpoints/rl_last.pt"
PHASE2_CKPT="runs/rl_phase2_mixed/run_001/checkpoints/rl_last.pt"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "=== STEP 1: Generación de datos IL ==="
python scripts/tournament/run.py
log "Datos generados."

log "=== STEP 2: Entrenamiento IL ==="
python train.py --config training/il_config.json
log "IL completado. Checkpoint: $IL_CKPT"

log "=== STEP 3: RL Fase 1 ==="
# Reemplaza <IL_BEST> en el config con la ruta real
sed -i "s|<IL_BEST>|$IL_CKPT|g" training/rl_phase1.json
python train.py --config training/rl_phase1.json
log "Fase 1 completada. Checkpoint: $PHASE1_CKPT"

log "=== STEP 4: RL Fase 2 ==="
sed -i "s|rl_phase1_anchored/run_001|rl_phase1_anchored/run_001|g" training/rl_phase2.json
python train.py --config training/rl_phase2.json
log "Fase 2 completada. Checkpoint: $PHASE2_CKPT"

log "=== STEP 5: RL Fase 3 ==="
python train.py --config training/rl_phase3.json
log "Fase 3 completada."

log "=== PIPELINE COMPLETO ==="
python train.py eval \
    --checkpoint runs/rl_phase3_league/run_001/checkpoints/rl_best_winrate.pt \
    --opponents heuristic.oracle_sniper \
    --n-matches 30
```

```bash
chmod +x scripts/run_pipeline.sh
nohup make pipeline > runs/pipeline.log 2>&1 &
echo "PID: $!"
```

---

## 8. Hoja de ruta completa (zero → Kaggle)

| Día | Paso | Acción | Tiempo estimado | Criterio de avance |
|---|---|---|---|---|
| 0 | Fix bugs | Bugs 1-4 de la Sección 0 + `make test-quick` | 1-2h | Tests verdes, dry-run OK |
| 0 tarde | Setup VM | Provisionar GCP, instalar, dry-run | 1h | CUDA visible, torch OK |
| 1 mañana | Generar datos | `make data` | 1-1.5h | ~1500 eps, < 3% draws |
| 1 tarde | IL training | `make train-bg` | 2-3h | Win rate ≥ 0.85 baseline, ≥ 0.40 sniper |
| 1-2 noche | RL Fase 1 | `make train-phase1` | ~5h | Win rate ≥ 0.65 sniper, ≥ 0.30 oracle |
| 2-3 | RL Fase 2 | `make train-phase2` | ~9h | Win rate ≥ 0.55 oracle (consistente) |
| 3-4 | RL Fase 3 | `make train-phase3` | ~12h | Win rate ≥ 0.70 oracle |
| 4 | Torneo + submit | 3-way tournament → `make submit-neural` | 1h | ELO de rl_phase3 más alto |

**Total: ~30-35h de cómputo, ~3-4 días de calendario.**

### Árbol de decisiones si algo falla

| Síntoma | Causa probable | Acción |
|---|---|---|
| IL: val_loss no baja | Datos pobres o pocos | +500 episodios oracle_sniper self-play |
| IL: win rate < 0.40 vs sniper tras 50 epochs | Dataset insuficiente o muy desbalanceado | Revisar distribución con DataCatalog, añadir datos |
| RL Fase 1: colapso (entropy < 0.3) | LR muy alto o KL insuficiente | Subir `kl_bc_coef_start` a 2.0 |
| RL Fase 1: no aprende (reward estancado) | `bc_policy_path` incorrecto o cache faltante | Verificar paths, recheck Bug 1 |
| RL Fase 2: regresión vs heurísticos | Pool de self-play demasiado débil | Volver a Fase 1 + 100 iters más, luego reiniciar Fase 2 |
| RL Fase 3: plateau pronto | Liga demasiado pequeña | Subir `max_snapshots` a 12, bajar `snapshot_every` a 25 |
| RL Fase 2: entropy collapse | Exploration insuficiente | Subir `entropy_coef_action_type` a 0.03 |

---

## 9. Parámetros clave — referencia rápida

### ¿Por qué estos coeficientes de entropía?

El juego tiene una cabeza de acción con 3 clases (NO_OP, LAUNCH, tercera no usada). La mayoría de planetas deben hacer NO_OP la mayoría del tiempo. Sin entropía suficiente en `action_type`, el modelo colapsa a NO_OP. Por eso `entropy_coef_action_type=0.02` es mayor que `target` y `amount` (0.005 cada uno). La entropía natural de `target` (hasta ~50 opciones, log 50 ≈ 3.9) ya es alta sin bonus extra.

### ¿Por qué `gamma=0.997` y no 0.99?

Con 500 turnos y `gamma=0.997`, el descuento a horizonte 500 es `0.997^500 ≈ 0.22` — la recompensa terminal sigue siendo visible al inicio. Con `gamma=0.99` sería `0.99^500 ≈ 0.007` — casi invisible. El horizonte efectivo con `gamma=0.997` es ~333 turnos, apropiado para este juego largo.

### ¿Por qué `reward_lambda=0.5`?

El shaping potencial es `0.5 * (0.997 * Phi(s') - Phi(s))`. Con `w_production=1.0`, cuando capturas un planeta de producción 3 sobre producción total 20, el shaping da `0.5 * (3/20) ≈ 0.075` — notable pero no dominante frente a los eventos (`r_event_capture_enemy=0.5`) ni al terminal (`r_terminal_win=10.0`). El balance es correcto.

---

## Apéndice — Archivos de config a crear

Resumen de los 3 nuevos JSONs de config RL (además de los existentes):

| Archivo | Fase | Iters | LR | self_play_prob | KL-BC |
|---|---|---|---|---|---|
| `training/rl_phase1.json` | Ancla IL | 300 | 3e-4 | 0.0 | sí (1.0→0.1, 200 iters) |
| `training/rl_phase2.json` | Self-play mixto | 500 | 3e-4 | 0.4 | no |
| `training/rl_phase3.json` | Liga | 700 | 1e-4 | 0.7 | no |

Y los dos scripts nuevos:
- `scripts/watch_run.sh` — monitorización en terminal
- `scripts/run_pipeline.sh` — pipeline completo automatizado
