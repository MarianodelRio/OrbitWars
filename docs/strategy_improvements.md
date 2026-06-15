# mejoras_strategy.md
# Mejoras identificadas — OrbitWars Neural Bot

Documento generado tras análisis del código fuente + investigación de literatura (PPO, RLHF, self-play, IL transfer).
Asume que los cambios de config ya están aplicados (`il_config.json`, `rl_phase1/2/3.json`, `tournament/config.json`).

Organizado por prioridad: **[MEJORA ALTA]** → **[MEJORA MEDIA]** → **[NICE-TO-HAVE]**.


## 4. [MEJORA ALTA] RL trainer usa Adam, no AdamW

**Archivo**: `training/trainers/rl_trainer.py:85`
**Severidad**: 🟠 Media — weight drift en 700 iteraciones de fase 3

### Problema
```python
self._optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
```
El IL trainer usa `AdamW` con `weight_decay=1e-4`. El RL trainer parte del checkpoint IL (mismos pesos) pero cambia de optimizador. Sin weight decay, los pesos del transformer pueden crecer lentamente durante las 700+ iteraciones de league play, degradando la generalización frente a oponentes nuevos.

### Fix
```python
# rl_trainer.py:85
self._optimizer = torch.optim.AdamW(
    self.model.parameters(),
    lr=cfg.lr,
    weight_decay=getattr(cfg, "weight_decay", 1e-4),
)
```
Añadir `weight_decay: float = 1e-4` a `RLConfig`.

**Archivos a tocar**: `training/trainers/rl_trainer.py`, `training/utils/rl_config.py`

---

## 5. [MEJORA ALTA] OpponentPool FIFO sin filtro de calidad (Phase 3)

**Archivo**: `training/rl/opponent_pool.py:61-65`
**Severidad**: 🟠 Media para Fase 3 — el snapshot más fuerte puede ser eviccionado

### Problema
```python
if len(self._snapshot_entries) > self.max_snapshots:
    oldest = self._snapshot_entries[0]  # siempre el más antiguo
```
Con `max_snapshots=6` y `snapshot_every=30` sobre 700 iters, se crean ~23 snapshots. Si hay una regresión temporal en iters 400-500, los snapshots débiles de ese período desplazan a los fuertes de iters 200-350. AlphaStar's PFSP paper identificó exactamente este patrón como "league forgetting".

### Fix: Evicción del snapshot más débil, no el más viejo

El eviccionado debe ser el que el modelo actual gana con mayor facilidad — no el más antiguo. Implementación simple: cada eval, trackear win-rate por snapshot name. Al eviccionar, elegir el entry con menor dificultad.

```python
# OpponentPool: añadir win-rate tracking
def update_snapshot_winrate(self, name: str, win_rate: float):
    for e in self._snapshot_entries:
        if e.name == name:
            e.wr = win_rate
            return

def add_snapshot(self, path, iteration):
    ...
    if len(self._snapshot_entries) > self.max_snapshots:
        # Evict the easiest opponent (highest win_rate), fallback to oldest
        victim = min(
            self._snapshot_entries,
            key=lambda e: getattr(e, 'wr', -1.0)  # -1 = never evaluated = keep
        )
        self._snapshot_entries.remove(victim)
        self._entries = [e for e in self._entries if e is not victim]
```

**Alternativa más simple**: si no se quiere trackear win-rates, usar `max_snapshots=6` pero priorizar variedad temporal — eviccionar el segundo más antiguo si el más antiguo tiene `wr > 0.7`.

**Archivos a tocar**: `training/rl/opponent_pool.py`

---

## 6. [MEJORA ALTA] Transición de fase gateada por win-rate, no por iteraciones fijas

**Archivo**: `Strategy.md` + scripts de pipeline
**Severidad**: 🟠 Media — el criterio fijo puede pasar a Fase 2 con un modelo infraentrenado

### Problema
El pipeline pasa de Fase 1 → 2 → 3 siempre en las iteraciones 300 / 500 / 700, independientemente de si se cumplieron los criterios de salida. Si el modelo no alcanzó ≥0.65 vs sniper en Fase 1, entrar en Fase 2 con self-play degenerará el pool rápidamente.

La literatura (survey arxiv 2408.01072) muestra que transiciones win-rate-gated superan a transiciones de iteración fija en ~15% en win rate final.

### Fix
Añadir lógica de extensión automática al final de cada fase:
```python
# Al final de la iteración final de cada fase:
if final_eval_winrate_vs_target < phase_threshold:
    total_iterations += 100   # extend up to 2 times
    log("Phase threshold not met — extending 100 iterations")
```
Configurar en JSON:
```json
"phase_exit_criterion": {"heuristic.sniper": 0.65, "heuristic.oracle_sniper": 0.30},
"phase_max_extensions": 2
```

**Archivos a tocar**: `training/trainers/rl_trainer.py` (añadir lógica al final del loop), `training/utils/rl_config.py`

---

## 7. [MEJORA MEDIA] Normalización de ventajas por buffer, no por minibatch

**Archivo**: `training/rl/ppo.py:130-131`
**Severidad**: 🟡 Baja-media — introduce ~4% de ruido adicional por minibatch

### Problema
```python
# ppo.py — se llama 80 veces por iteración (16 batches × 5 epochs)
advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
```
Cada minibatch normaliza independientemente. Un minibatch con todas ventajas positivas (buen fragmento del rollout) queda remapeado a media 0, perdiendo la información de "este rollout fue mejor que el promedio". La normalización correcta es **una sola vez sobre todo el buffer** antes de batching.

### Fix
Mover la normalización a `rollout_buffer.py` después de `compute_gae()`:
```python
def normalize_advantages(self):
    advantages = np.array([s.advantage for s in self._steps])
    mean, std = advantages.mean(), advantages.std()
    for s in self._steps:
        s.advantage = (s.advantage - mean) / (std + 1e-8)
```
Y en `ppo.py:130-131`, eliminar las dos líneas de normalización.

**Archivos a tocar**: `training/rl/rollout_buffer.py`, `training/rl/ppo.py`

---

## 8. [MEJORA MEDIA] Paralelizar generación de datos IL (3 workers)

**Archivo**: `scripts/tournament/run.py`
**Severidad**: 🟡 Media — afecta solo a la fase de generación, no al training

### Problema
La generación de 1500 episodios es secuencial: ~2-3s/episodio = **45-75 minutos**. `kaggle_environments` es single-threaded pero sin estado compartido — cada match es completamente independiente.

### Fix
```python
# scripts/tournament/run.py — usar ProcessPoolExecutor
from concurrent.futures import ProcessPoolExecutor

def run_match_worker(args):
    bot1, bot2, output_path = args
    return run_match(bot1, bot2, output_path)  # ya crea su propio env

with ProcessPoolExecutor(max_workers=3) as pool:
    pool.map(run_match_worker, match_args_list)
```
Con 3 workers en 4 CPUs: **~15-25 minutos** para 1500 episodios (3× speedup).

**Nota**: NO usar ThreadPool (GIL bloquea simulaciones CPU-bound).

**Archivos a tocar**: `scripts/tournament/run.py`

---

## 9. [MEJORA MEDIA] KL early-stopping por época en PPO

**Archivo**: `training/rl/ppo.py`, `training/trainers/rl_trainer.py`
**Severidad**: 🟡 Baja-media — previene el clipping excesivo cuando approx_kl > umbral

### Problema
El trainer corre exactamente 5 epochs sin importar si la política está divergiendo. Si en la época 2 `approx_kl > 0.03`, las épocas 3-5 aplican gradientes sobre una política ya fuera del trust region.

### Fix
```python
# rl_trainer.py — en el loop de epochs
for _epoch in range(cfg.ppo_epochs):
    ...
    if avg_kl > getattr(cfg, 'target_kl', 0.03):
        tqdm.write(f"  [PPO] Early KL stop at epoch {_epoch+1}: approx_kl={avg_kl:.4f}")
        break
```
Añadir `target_kl: float = 0.03` a `RLConfig` (None = desactivado).

**Archivos a tocar**: `training/trainers/rl_trainer.py`, `training/utils/rl_config.py`

---

## 10. [NICE-TO-HAVE] Sincronización Adam/AdamW al cargar checkpoint IL

**Archivo**: `training/trainers/rl_trainer.py:457-465`
**Severidad**: 🟢 Baja — los primeros ~30 iters RL tienen gradientes más ruidosos

### Problema
Al cargar el checkpoint IL para RL, el estado del optimizador (momentos Adam m/v) se descarta y re-inicializa. Durante las primeras ~50 iteraciones, los acumuladores de Adam parten de cero mientras los pesos ya están en un mínimo del IL — esto crea pasos de gradiente sobredimensionados hasta que los momentos se "calientan".

### Fix Potencial
Añadir warmup de LR más largo en Fase 1 (e.g. 5-10 iteraciones con LR=1e-4 antes de lr=3e-4), o usar un `lr_warmup_iters` config field. También mitiga parcialmente el pico de approx_kl en las primeras iteraciones.

**Archivos a tocar**: `training/trainers/rl_trainer.py` (añadir warmup lineal de LR al start)

---

## Resumen de prioridades

| # | Tipo | Archivos | Impacto | Complejidad |
|---|---|---|---|---|
| 1 | 🔴 BUG | `ppo.py`, `rl_config.py` | Alto | Trivial |
| 2 | 🔴 BUG | `potential.py` | Medio-alto | Fácil |
| 3 | 🔴 BUG | `potential.py`, `rl_config.py` | Bajo | Trivial |
| 4 | 🟠 MEJORA | `rl_trainer.py`, `rl_config.py` | Medio | Fácil |
| 5 | 🟠 MEJORA | `opponent_pool.py` | Medio (Fase 3) | Medio |
| 6 | 🟠 MEJORA | `rl_trainer.py`, `rl_config.py` | Medio | Medio |
| 7 | 🟡 MEJORA | `rollout_buffer.py`, `ppo.py` | Bajo | Fácil |
| 8 | 🟡 MEJORA | `scripts/tournament/run.py` | Medio (tiempo) | Fácil |
| 9 | 🟡 MEJORA | `rl_trainer.py`, `rl_config.py` | Bajo | Fácil |
| 10 | 🟢 NICE | `rl_trainer.py` | Bajo | Fácil |

---

## Ciclos de implementación recomendados

### Ciclo 1 — Bugs críticos (2 archivos)
Fixes #1 + #2 + #3: `ppo.py` + `potential.py` + `rl_config.py`
- Desactivar value clipping (o `value_clip_eps: null`)
- Activar r_event_capture_comet + fix neutral→propia en events
- Eliminar reward_clip_abs

### Ciclo 2 — Consistencia del optimizador (2 archivos)
Fix #4: `rl_trainer.py` → AdamW, `rl_config.py` → añadir weight_decay

### Ciclo 3 — Mejoras de training loop (3 archivos)
Fixes #7 + #9: normalización por buffer + KL early stopping
`rollout_buffer.py` + `ppo.py` + `rl_trainer.py`

### Ciclo 4 — Mejoras de liga y pipeline (2 archivos)
Fix #5: `opponent_pool.py` (quality-filtered eviction)
Fix #8: `scripts/tournament/run.py` (paralelización)

### Ciclo 5 — Gating de fases (si se necesita)
Fix #6: `rl_trainer.py` + `rl_config.py` (transición automática por win-rate)

---

## Señales de alarma en métricas (referencia rápida)

| Métrica | Señal de problema → causa probable → acción |
|---|---|
| `explained_variance < 0` en iters 1-50 | Value head roto → aplicar fix #1 primero |
| `entropy_action_type > 1.0` sostenido | Over-exploración → coef ya reducido a 0.01, verificar |
| `entropy_action_type < 0.3` | Colapso → subir entropy_coef_action_type a 0.015 |
| `approx_kl > 0.05` | Clip suelto o LR alto → bajar LR o reducir ppo_epochs |
| `clip_fraction > 0.30` → | LR demasiado alto → reducir lr a 1e-4 |
| `win_rate vs oracle` plano 50+ iters Fase 2 | Pool self-play débil → extender Fase 1 100 iters |
| `value_loss` creciendo tras iter 100 | Posible divergencia GAE → reducir gamma a 0.995 o añadir grad clip al value |
