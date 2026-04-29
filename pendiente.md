# Pendientes

## 1. Integrar `relational_tensor` en IL training (opción B — producción)

**Contexto:** El cache HDF5 (`data/cache/il_planet_policy.h5`) no incluye `relational_tensor`. El IL trainer llama al modelo sin este tensor, por lo que los pesos de `rel_proj` nunca reciben gradientes. En RL se pasa `relational_tensor` real → NaN inmediato.

**Tareas:**
- Añadir `relational_tensor` al schema del cache HDF5 en `bots/neural/training.py`:
  - En `build_il_cache`: guardar `relational_tensor` con shape `(N, max_planets, max_planets, 4)` float32
  - En `PrecomputedILDataset._load_buffer`: cargar `relational_tensor` del buffer
  - En `PrecomputedILDataset.__getitem__`: devolver `relational_tensor` en el dict
  - Bump `schema_version` de 3 → 4
- En `training/trainers/il_trainer.py`: pasar `relational_tensor` al modelo en train y val loops
- Reconstruir cache (`make cache`) y reentrenar IL (`make train`)
- En `training/trainers/rl_trainer.py`: revertir fix temporal de `rt=None` y volver a pasar `rt` real
- En `training/rl/ppo.py`: revertir `None` → `batch["relational_tensor"]` en los dos calls al modelo (línea ~128 y bc_model ~174)
- **No revertir** el fix de `planet_policy_model.py` Stage 2 (pre-combined mask): es una mejora correcta independiente del relational_tensor

**Nota de performance:** `_build_relational_tensor` usa doble bucle Python O(n²). Para 1500 episodios (~1.5M pasos) tardará ~90 min. Vectorizar con numpy antes de reconstruir el cache de producción.

---

## 2. Vectorizar `_build_relational_tensor` (performance)

**Archivo:** `bots/neural/state_builder.py`, método `_build_relational_tensor`

**Problema:** Doble bucle Python puro sobre `n_planets²` → ~3 min para 50 episodios, ~90 min para 1500.

**Solución:** Reescribir con operaciones numpy vectorizadas:
- `dists`: ya vectorizado en `_build` (línea 198). Reutilizar.
- `angle_diff`: usar `np.arctan2` + broadcasting sobre matrices `(n, n)`.
- `same_owner`: outer product de `owners == owners[:, None]`.
- `reachable`: `dists <= 6.0 * 50.0`.

Resultado esperado: reducción de ~100x en tiempo de cache build.

---

## 3. Arreglar pérdida de output en logs

**Problema:** Los logs de entrenamiento (`runs/il_train.log`, `runs/rl_phase*.log`) no muestran output en tiempo real — el buffer de stdout de Python acumula hasta 8KB antes de escribir.

**Soluciones a evaluar:**
- Opción A (Makefile): añadir `-u` flag a todos los comandos `nohup python ...` en el `Makefile` (líneas `train-bg`, `train-phase1/2/3`)
- Opción B (código): añadir `flush=True` a los `print(...)` clave en `ILTrainer` y `RLTrainer`
- Opción A es la más limpia — un cambio en el Makefile cubre todo.

**Archivos afectados:** `Makefile` (líneas 80, 85, 89, 93)
