# Mejoras de Performance — Orbit Wars Training Pipeline

**Hardware objetivo:** 4 CPUs @ 2.2 GHz · 16 GB RAM · NVIDIA L4 (24 GB VRAM, tensor cores bf16)

**Diagnóstico central:** el loop RL es *rollout-bound*, no *GPU-bound*. La L4 permanece >90% idle durante la recolección de experiencias porque `kaggle_environments` es single-threaded por episodio y el código actual no usa paralelismo de ningún tipo. El PPO update sí usa la GPU, pero también corre en fp32 a pesar de que la L4 tiene tensor cores optimizados para bfloat16.

---

## Índice de mejoras

| # | Mejora | Archivos | Speedup RL total | Esfuerzo |
|---|--------|----------|------------------|----------|
| C | [bf16 AMP para RL](#c--bf16-amp-para-rl) | 2 | 10–15% | muy bajo |
| B | [Buffer pre-tensorizado](#b--buffer-pre-tensorizado) | 2 | 5–10% + ahorro RAM | medio |
| D | [Evaluación paralela](#d--evaluación-paralela) | 1–2 | 2–6% | medio |
| F | [DataLoader workers + pin_memory](#f--dataloader-workers--pin_memory) | 1–2 | 3–7% | muy bajo |
| A | [Rollout paralelo (SubprocVecEnv)](#a--rollout-paralelo-subprocvecenv) | 4–5 | **2–2.5x** (el mayor) | alto, multi-ciclo |
| G | [Precarga eager de snapshots](#g--precarga-eager-de-snapshots) | 1 | 1–3% | trivial |
| H | [Tensor construction en rollout loop](#h--tensor-construction-en-rollout-loop) | 1 | 2–5% | trivial |
| E | [Generación de datos IL paralela](#e--generación-de-datos-il-paralela) | 1–2 | one-shot | bajo |

**Secuencia recomendada:** C → B → D → A (A requiere diseño multi-ciclo independiente)

---

## C — bf16 AMP para RL

### Qué hace hoy el código

`compute_ppo_loss` en [training/rl/ppo.py](training/rl/ppo.py) y el forward pass del rollout en [training/trainers/rl_trainer.py:202-212](training/trainers/rl_trainer.py#L202) corren íntegramente en **fp32**. Cada matmul del transformer (proyecciones Q/K/V, FFN, LSTM) usa FP32 en todos los casos.

En cambio, el IL trainer en [training/trainers/il_trainer.py:344-402](training/trainers/il_trainer.py#L344) ya usa `torch.autocast(device_type="cuda", dtype=torch.bfloat16)` correctamente.

### Por qué es importante en la L4

La L4 tiene tensor cores de tercera generación con soporte nativo para bfloat16. En bf16, los matmuls se ejecutan en unidades de hardware dedicadas que son **2–3× más rápidas** que los CUDA cores fp32. bfloat16 tiene el mismo rango de exponente que fp32 (8 bits), con menos mantisa (7 bits vs 23 bits), lo que lo hace numéricamente seguro para redes neuronales sin necesidad de GradScaler.

### Qué cambiar

**Archivos:** `training/rl/ppo.py` y `training/trainers/rl_trainer.py`

#### En `ppo.py` — `compute_ppo_loss`

Envolver **solo el forward del modelo** en autocast. El ratio, el value clipping y el KL-to-BC deben quedar fuera en fp32:

```
# DENTRO de autocast — el modelo
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    output, _ = model(planet_features, fleet_features, fleet_mask,
                      global_features, planet_mask, rt)
    if bc_model is not None and kl_bc_coef > 0.0:
        bc_out, _ = bc_model(...)   # BC forward también dentro

# FUERA de autocast — aritmética sensible
ratio = torch.exp(new_log_prob - log_prob_old)   # fp32 — crítico
v_clipped = value_old + torch.clamp(...)          # fp32
kl_at = F.kl_div(...)                            # fp32
```

#### En `rl_trainer.py` — rollout forward y bootstrap

```
# Rollout step (línea ~212)
with torch.no_grad():
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output, new_hidden = self.model(pf, ff, fm, gf, pm, rt, hidden)

# Bootstrap value (línea ~293)
with torch.no_grad():
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output, _ = self.model(pf, ff, fm, gf, pm, rt, hidden)

# IL distillation forward (línea ~332)
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    il_output, _ = self.model(il_pf, il_ff, il_fm, il_gf, il_pm, il_rt)
```

### Guardas de correctness obligatorias

1. **`torch.exp(new_log_prob - log_prob_old)` debe quedar en fp32.** En bf16, la diferencia de log-probs tiene menos precisión en la mantisa; al exponenciar, un error de ±0.01 en bf16 produce ratios fuera de la zona de clip y explota el gradiente. Salir del contexto autocast antes de esta línea es suficiente.

2. **El bloque KL-to-BC ([ppo.py:178-223](training/rl/ppo.py#L178)) debe ser fp32.** `F.kl_div` con `-inf` en los logits mascarados produce NaN en bf16 con más frecuencia que en fp32. El forward del modelo con `bc_model` puede estar en autocast; la aritmética KL no.

3. **Ventaja (`advantage`) y retorno (`ret`) almacenados en buffer son fp32** — no tocar esos tensores dentro de autocast.

4. **Tests**: cualquier test que afirme valores exactos de loss necesita añadir tolerancia (`atol=1e-2`). bf16 no es bit-exact respecto a fp32.

### Ganancia esperada

- PPO update: **20–35% más rápido** en la L4 (los matmuls del transformer usan tensor cores).
- Rollout forward (B=1): **10–20% más rápido** (latency-bound, no compute-bound, pero aun así mejora).
- **Total por iteración RL: 10–15% de speedup**.

---

## B — Buffer pre-tensorizado

### Qué hace hoy el código

`RolloutBuffer` en [training/rl/rollout_buffer.py](training/rl/rollout_buffer.py) almacena una lista Python de `RolloutStep` dataclass. Cada dataclass contiene un `dict` con los arrays numpy del estado, además de máscaras, tensores LSTM h_n/c_n, escalares de reward, etc.

`get_batches()` (línea 130) reconstruye tensores así por cada minibatch de cada época PPO:

```python
planet_features = torch.tensor(
    np.stack([s.state["planet_features"] for s in batch_steps]),
    dtype=torch.float32,
    device=device,
)
```

Con 2048 steps, 4 épocas PPO y ~8 minibatches por época, esto llama a `np.stack` **unas 64 veces** por iteración sobre listas de objetos Python. Cada llamada itera la lista, extrae el array, los apila en un nuevo array de C contiguo, y lo convierte a tensor. Mucho overhead para datos que ya están en memoria.

### La solución

Pre-alocar arrays numpy contiguos de forma `(capacity, ...)` al crear el buffer. `add()` escribe directamente en el índice del cursor. `get_batches()` genera los batches con un simple slice + `.to(device)`.

```
buffer._planet_features    # shape (2048, 50, 24)   float32
buffer._fleet_features     # shape (2048, 200, 16)  float32
buffer._relational_tensor  # shape (2048, 50, 50, D) float32
buffer._action_types       # shape (2048, 50)        int64
buffer._advantages         # shape (2048,)           float32
buffer._h_n                # shape (2048, 1, 1, G)   float32 — LSTM
...
```

`get_batches()` hace `np.take(indices, axis=0)` o simplemente indexa con fancy indexing, luego hace una sola transferencia CPU→GPU por campo:

```python
planet_features = torch.from_numpy(self._planet_features[batch_indices]).to(device, non_blocking=True)
```

### Beneficios colaterales

- **RAM**: la lista de dataclasses Python tiene overhead de objeto (~200 bytes por `RolloutStep`). Con 2048 steps, son ~400 KB de objetos Python además de los datos. El buffer pre-tensorizado elimina ese overhead.
- **Caché de CPU**: los datos están en arrays C-contiguos de memoria densa. El acceso por índices aleatorios (batch shuffle) tiene mejor localidad que desreferenciar punteros de lista Python.
- **Más sencillo ampliar a rollout paralelo** (Opción A): si los datos ya son arrays numpy contiguos, hacer merge de buffers de distintos workers es un simple `np.concatenate` al final.

### Restricción importante (16 GB RAM)

Calcular el tamaño del buffer antes de implementar:

```
planet_features:    2048 × 50 × 24 × 4 bytes = ~9.8 MB
fleet_features:     2048 × 200 × 16 × 4 bytes = ~26 MB
relational_tensor:  2048 × 50 × 50 × D × 4 bytes   ← comprobar D
global_features:    2048 × G_dim × 4 bytes
LSTM h_n + c_n:     2048 × 1 × 1 × 384 × 4 bytes × 2 = ~6 MB
actions (int64):    2048 × 50 × 3 × 8 bytes = ~2.5 MB
scalars (float32):  2048 × 7 campos × 4 bytes = ~0.06 MB
```

Total estimado: **~60–100 MB** dependiendo del tamaño del relational_tensor. Perfectamente dentro de 16 GB.

### Archivos afectados

- `training/rl/rollout_buffer.py` — reescritura de `RolloutBuffer`
- `training/trainers/rl_trainer.py` — adaptación del `RolloutStep` que se pasa a `buffer.add()`

---

## D — Evaluación paralela

### Qué hace hoy el código

`Evaluator.run()` en [training/evaluation/evaluator.py](training/evaluation/evaluator.py) itera los oponentes y los matches en serie:

```python
for opp_name in self._opponents:   # 4 oponentes en fase 1
    raw = evaluate(neural_fn, opp_fn, n_matches=20)  # 20 partidas en serie
```

Con 4 oponentes × 20 partidas × ~3 s/partida = **~240 s de evaluación**. Esto bloquea el training loop.

### La solución

Usar `concurrent.futures.ProcessPoolExecutor` para correr los bloques de `(oponente, n_matches)` en paralelo. Con 4 CPUs se pueden correr 4 oponentes simultáneamente, reduciendo el tiempo de evaluación a ~60 s (1/4 del original).

Diseño del worker:

```python
def _eval_worker(args):
    checkpoint_path, opp_name, n_matches, run_dir = args
    # Cargar el bot desde checkpoint (no picklear el modelo vivo)
    bot = NeuralBot.load(checkpoint_path)
    neural_fn = make_agent(bot)
    opponent_fn = resolve(opp_name)
    return evaluate(neural_fn, opponent_fn, n_matches=n_matches)
```

El modelo se **carga desde disco en el worker** en lugar de picklarse desde el proceso padre. Esto evita los problemas de serialización de tensores CUDA con multiprocessing.

### Consideraciones Windows

En Windows, `multiprocessing` usa `spawn` por defecto (no `fork`). Cada worker arranca un intérprete Python nuevo, importa todos los módulos, y luego ejecuta el worker. El tiempo de arranque (~2–3 s por worker) se amortiza si `n_eval_matches >= 10`. Con `max_workers=4` y 4 oponentes, el overhead de spawn es despreciable frente a las 20 partidas.

`start_method='spawn'` es obligatorio en Windows — no hay elección.

### Archivos afectados

- `training/evaluation/evaluator.py` — añadir rama paralela en `run()`

---

## F — DataLoader workers + pin_memory

### Qué hace hoy el código

El `DataLoader` de IL distillation en [training/trainers/rl_trainer.py:163-169](training/trainers/rl_trainer.py#L163) se crea así:

```python
self._il_loader = DataLoader(
    il_dataset,
    batch_size=cfg.ppo_batch_size,
    shuffle=True,
    num_workers=0,       # ← lectura HDF5 bloqueante en main thread
    drop_last=True,
)
```

Con `num_workers=0`, cada acceso a `PrecomputedILDataset.__getitem__` hace un read de HDF5 en el thread principal durante el PPO update. Esto bloquea el forward pass mientras espera I/O de disco.

El IL trainer ([il_trainer.py:204-213](training/trainers/il_trainer.py#L204)) tiene los mismos defaults con `num_workers` leído del config, pero si el config tiene `num_workers: 0` (que es el caso) el efecto es el mismo.

### La solución

**Para el RL trainer** (IL distillation DataLoader):

```python
self._il_loader = DataLoader(
    il_dataset,
    batch_size=cfg.ppo_batch_size,
    shuffle=True,
    num_workers=1,        # 1 worker es suficiente — no robar CPUs al rollout
    pin_memory=True,      # tensores en pinned memory → transferencia async a GPU
    persistent_workers=True,  # no respawnear por cada iteración
    prefetch_factor=2,
    drop_last=True,
)
```

**Para el IL trainer** en `il_config.json`:

```json
{
  "num_workers": 2,
  "pin_memory": true,
  "persistent_workers": true
}
```

### Condición de correctness: HDF5 fork-safe

`PrecomputedILDataset` debe abrir el archivo HDF5 **dentro de `__getitem__` o en `worker_init_fn`**, no en `__init__`. Si el file handle se hereda del proceso padre al hacer spawn/fork, el acceso concurrente al mismo `h5py.File` objeto produce corrupción de datos.

Patrón seguro:

```python
class PrecomputedILDataset:
    def __init__(self, path):
        self._path = str(path)
        self._h5 = None   # NO abrir aquí

    def _get_h5(self):
        if self._h5 is None:
            self._h5 = h5py.File(self._path, "r")  # Abrir al primer acceso
        return self._h5

    def __getitem__(self, idx):
        return self._get_h5()[...]  # Lazy open — cada worker abre su propio handle
```

### Limitación de CPUs

Con 4 CPUs y el rollout usando 1 de ellas, hay 3 libres. Usar `num_workers=1` o `num_workers=2` para DataLoader es conservador y seguro. No usar más de 2 para no privar de CPU al rollout o al PPO update.

### Archivos afectados

- `training/trainers/rl_trainer.py` — `DataLoader` de IL distillation
- `bots/neural/training.py` — verificar que `PrecomputedILDataset.__init__` no abre HDF5

---

## A — Rollout paralelo (SubprocVecEnv)

> **Nota:** Esta mejora tiene el mayor impacto potencial (2–2.5× speedup total RL) pero requiere **diseño multi-ciclo independiente**. No entra en un ciclo plan→implement→review de 3 archivos. Se documenta aquí para planificación futura.

### El cuello de botella real

El pipeline RL actual tiene esta estructura temporal por iteración:

```
[████████████ ROLLOUT (CPU, 1 core) ████████████][█ PPO update (GPU) █]
```

2048 steps de rollout × ~0.06 s/step (kaggle env step + state_builder + forward B=1) ≈ **~120 s de rollout**.
PPO update (4 épocas × 8 batches × forward+backward) ≈ **~15–20 s en fp32, ~10–12 s en bf16**.

La L4 está ociosa el 85% del tiempo esperando que el CPU genere experiencias.

### La solución: N workers paralelos

```
Worker 0: [env0 → obs → forward → step] × (2048/N)
Worker 1: [env1 → obs → forward → step] × (2048/N)
Worker 2: [env2 → obs → forward → step] × (2048/N)
Worker 3: [env3 → obs → forward → step] × (2048/N)
                                              ↓ merge buffers
Main:                                    [PPO update (GPU)]
```

Con N=3 workers (dejando 1 CPU al proceso principal), el rollout se comprime a **~40 s** → iteración total de ~50–52 s en lugar de ~130–140 s.

### Decisiones de diseño que requieren planificación

**1. Arquitectura de los workers**

Hay dos modelos posibles:

- **Worker actor (modelo local):** cada worker tiene una copia del modelo, hace forwards localmente, genera una trayectoria completa y la envía al main. Ventaja: sin overhead de IPC por step. Desventaja: el modelo en los workers se desactualiza entre iteraciones — hay que broadcastear los pesos nuevos al final de cada PPO update.

- **Worker env (main hace forwards):** los workers solo corren el env y envían observaciones; el main hace el forward en batch y devuelve acciones. Ventaja: pesos siempre frescos. Desventaja: mucho IPC por step (latencia por round-trip).

Para este juego, **worker actor** es mejor porque los episodios son largos (500 steps) y el forward B=1 en CPU es barato. El coste de sincronizar pesos una vez por iteración es insignificante.

**2. LSTM hidden state**

El hidden state `(h_n, c_n)` debe mantenerse por worker de forma independiente. Al resetear un episodio, el worker reinicia su propio hidden. El main process **nunca mezcla hidden states entre workers**. Esto es correcto porque PPO trata cada step como independiente después de la recolección.

**3. Sincronización de snapshots**

Cuando el main process guarda un nuevo snapshot en `OpponentPool`, debe notificar a todos los workers. La solución más simple: al principio de cada episodio nuevo, el worker consulta un archivo de control (`snapshots_manifest.json`) para ver si hay snapshots nuevos que cargar.

**4. `kaggle_environments` con múltiples procesos**

`kaggle_environments.make("orbit_wars")` tiene estado a nivel de módulo Python. En Windows (spawn), cada worker es un proceso nuevo limpio, por lo que no hay contaminación entre ellos. En Linux (fork), habría que hacer el `make()` **después** del fork, no antes.

**5. IPC y overhead**

Con `multiprocessing` de Python, la serialización de las trayectorias (lista de arrays numpy) puede ser el cuello de botella si se hace naively via pickle. Usar `shared_memory` de Python 3.8+ para los arrays del buffer hace la transferencia casi instantánea (zero-copy).

### Archivos afectados (multi-ciclo)

- `training/trainers/rl_trainer.py` — orquestación de workers, merge de buffers, broadcast de pesos
- `training/rl/rollout_buffer.py` — merge de múltiples buffers
- `training/envs/orbit_env.py` — debe ser instanciable desde un worker process
- `training/rl/opponent_pool.py` — snapshot manifest para sincronización
- `training/rl/vec_env.py` *(nuevo)* — abstracción del pool de workers

---

## G — Precarga eager de snapshots

### Qué hace hoy el código

`OpponentPool.get_agent()` en [training/rl/opponent_pool.py:18-21](training/rl/opponent_pool.py#L18) carga el modelo lazy en el primer uso:

```python
def get_agent(self):
    if self._agent is None:
        self._agent = self.loader()  # NeuralBot.load() desde disco
    return self._agent
```

Esto significa que la primera vez que un snapshot es sorteado durante el rollout, el training loop se bloquea mientras carga los pesos desde disco (~0.5–1 s). Este spike aparece aleatoriamente en medio del rollout y distorsiona las métricas de tiempo.

Además, `_current_model_as_agent()` en [rl_trainer.py:432](training/trainers/rl_trainer.py#L432) crea un `NeuralBot` wrapper nuevo **en cada episodio de self-play**. Este wrapper es barato de crear, pero la llamada a `make_agent()` implica ciertas inicializaciones que podrían cachearse.

### La solución

**En `add_snapshot()`**, llamar a `loader()` inmediatamente y cachear el resultado:

```python
def add_snapshot(self, path: Path, iteration: int) -> None:
    entry = PoolEntry(name=f"snapshot:{iteration}", loader=loader)
    entry._agent = entry.loader()  # Cargar ahora, no en el primer uso
    ...
```

**Para self-play**, cachear el wrapper entre episodios siempre que los pesos no hayan cambiado (no hayan corrido épocas PPO entre rollout y self-play sample).

### Archivos afectados

- `training/rl/opponent_pool.py`

---

## H — Tensor construction en rollout loop

### Qué hace hoy el código

Dentro del hot loop de `_collect_rollout` ([rl_trainer.py:202-207](training/trainers/rl_trainer.py#L202)), en cada uno de los 2048 steps se crean 6 tensores a partir de arrays numpy:

```python
pf = torch.tensor(state["planet_features"], dtype=torch.float32).unsqueeze(0).to(device)
ff = torch.tensor(state["fleet_features"],  dtype=torch.float32).unsqueeze(0).to(device)
fm = torch.tensor(state["fleet_mask"],      dtype=torch.bool).unsqueeze(0).to(device)
gf = torch.tensor(state["global_features"], dtype=torch.float32).unsqueeze(0).to(device)
pm = torch.tensor(state["planet_mask"],     dtype=torch.bool).unsqueeze(0).to(device)
rt = torch.tensor(state["relational_tensor"],dtype=torch.float32).unsqueeze(0).to(device)
```

`torch.tensor()` hace una **copia** del array numpy. `torch.from_numpy()` hace una vista sin copia (zero-copy). Para arrays de `state_builder` que ya tienen el dtype correcto, esto es trabajo innecesario.

Además, `.to(device)` hace 6 transferencias CPU→GPU independientes. Una sola llamada a `torch.cat` o empaquetar en un dict y transferir de una vez sería más eficiente.

### La solución

```python
# Reemplazar torch.tensor → torch.from_numpy para arrays ya en el dtype correcto
pf = torch.from_numpy(state["planet_features"]).unsqueeze(0).to(device, non_blocking=True)
ff = torch.from_numpy(state["fleet_features"]).unsqueeze(0).to(device, non_blocking=True)
# etc.
```

`non_blocking=True` permite que la transferencia CPU→GPU sea asíncrona si los tensores están en pinned memory. Con `pin_memory` activado en la construcción del estado (posible extensión futura), esto elimina la sincronización.

Nota: `torch.from_numpy` requiere que el array numpy sea **C-contiguo** y del dtype correcto. Si `state_builder` devuelve arrays con dtype incorrecto o no contiguos, usar `.ascontiguousarray()` antes o mantener `torch.tensor`.

### Archivos afectados

- `training/trainers/rl_trainer.py` — las 6 líneas de construcción de tensor en `_collect_rollout`

---

## E — Generación de datos IL paralela

### Qué hace hoy el código

`make data` lanza un único proceso Python que genera los 1500 episodios de IL en serie. Con ~2–3 s/episodio, la generación completa tarda **50–75 minutos**. No hay paralelismo aunque haya 4 CPUs disponibles.

### La solución

Lanzar 4 procesos Python independientes, cada uno generando ~375 episodios, cada uno escribiendo su propio HDF5 shard. Al terminar todos, el cache builder de IL consolida los shards.

```bash
# Ejemplo: 4 procesos en paralelo
python scripts/generate_data.py --n_matches 95 --seed 0 --output data/shard_0.h5 &
python scripts/generate_data.py --n_matches 95 --seed 1 --output data/shard_1.h5 &
python scripts/generate_data.py --n_matches 95 --seed 2 --output data/shard_2.h5 &
python scripts/generate_data.py --n_matches 95 --seed 3 --output data/shard_3.h5 &
wait
python scripts/merge_shards.py data/shard_*.h5 data/il_full.h5
```

**Speedup esperado:** ~3.5× (4 procesos, overhead de merge mínimo).
**Tiempo estimado:** 15–20 minutos en lugar de 60–75 minutos.

### Consideraciones

- Cada proceso debe tener **semilla distinta** (`--seed N`). Si todas usan la misma semilla, `random.choice` de bots y mapas produce los mismos episodios duplicados.
- `kaggle_environments` puede tener estado global que impida múltiples instancias en el mismo proceso. En procesos separados (no threads) no hay problema.
- Esta es una **mejora one-shot** (solo se ejecuta al generar datos, no durante el training loop). No afecta a la velocidad de iteración RL.

### Archivos afectados

- `scripts/` — script de generación con argumento `--seed` y `--output`
- `dataset/` — función de merge de shards HDF5 (probablemente ya soportado por `DataCatalog.scan()` si los shards están en el directorio correcto)

---

## Interacciones entre mejoras

| Combinación | Interacción |
|-------------|-------------|
| **C + A** | Completamente complementarias. C acelera el GPU update; A acelera la CPU rollout. Atacan cuellos de botella distintos. |
| **B + A** | Complementarias y sinérgicas. Pre-tensorizar el buffer hace trivial hacer merge de múltiples buffers de workers. B es prerequisito recomendado antes de A. |
| **B + C** | Sin conflicto. Buffer en fp32 → autocast alrededor del forward solamente → correcto. No pre-alocar en bf16. |
| **F + A** | Conflicto de CPUs. Si A usa N=3 workers, F no debe usar más de 1 worker DataLoader. Total: 4 procesos Python sin sobrepasar los 4 CPUs físicos. |
| **D + A** | Conflicto temporal menor: la evaluación paralela no debería correr al mismo tiempo que el rollout paralelo porque ambos saturarían los CPUs. Secuenciar eval solo al terminar la iteración de rollout (ya es así en el código actual). |
| **G + A** | G se vuelve más importante con A: si hay 3 workers que pueden samplear snapshots, la carga lazy dispararía 3 bloqueos simultáneos en el primer uso. Eager loading elimina el problema. |

---

## Resumen del impacto acumulado

Aplicando las mejoras en secuencia:

| Después de... | Speedup acumulado (iteración RL) |
|---------------|----------------------------------|
| Solo C (bf16 AMP) | ~1.12× |
| C + B (buffer) | ~1.20× |
| C + B + D (eval paralela) | ~1.25× |
| C + B + D + A (rollout paralelo) | **~2.5–3×** |

El paso de C→B→D es incremental y bajo riesgo. El paso final (A) es el salto grande pero requiere un ciclo de diseño dedicado.

---

*Generado: 2026-04-29 — basado en análisis del código en commit actual (branch: main)*
