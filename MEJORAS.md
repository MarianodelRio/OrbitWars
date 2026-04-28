# Mejoras potenciales del bot

Análisis basado en la arquitectura actual: `PlanetPolicyModel` (69K params), `StateBuilder`, `shaped_reward`.

---

## 1. Representación del estado

### 1a. Distancia entre planetas (falta relación pairwise)
**Problema actual:** cada planeta se representa con features absolutas (x/100, y/100). El modelo no sabe explícitamente que el planeta A está a 5 unidades del planeta B y el planeta C a 40.

**Mejora:** añadir features relativas por par (source, target):
- distancia normalizada entre cada planeta mío y cada enemigo/neutral
- ángulo relativo entre ellos

Esto es especialmente crítico para el pointer network de target selection — actualmente el modelo tiene que inferir distancias desde coordenadas absolutas.

### 1b. Flota en tránsito "amenaza recibida"
**Problema actual:** las flotas enemigas en vuelo existen en `fleet_features`, pero el modelo no sabe explícitamente qué planeta están atacando ni cuándo llegan.

**Mejora:** para cada planeta, añadir:
- `incoming_enemy_ships`: suma de ships de flotas enemigas dirigidas a ese planeta
- `turns_to_arrival`: pasos hasta que llega la flota más cercana
- `incoming_friendly_ships`: reinforcements propios en camino

Esto permite al modelo aprender a defender.

### 1c. Velocidad angular del planeta
**Problema actual:** los planetas orbitales se mueven 0.025–0.05 rad/turn. La posición actual se normaliza pero no hay información de hacia dónde se mueve el planeta.

**Mejora:** añadir `sin(angular_velocity * turns_remaining)` y `cos(...)` como feature de cada planeta orbital. El modelo podría aprender a anticipar posiciones futuras.

### 1d. Global features limitadas
**Problema actual:** solo 4 features globales: turn, ship_ratio, planet_ratio, fleet_count.

**Mejora candidata:** añadir production_ratio (mi producción / total) — es mejor proxy de ventaja económica a largo plazo que planet_ratio puro, porque un planeta con production=5 vale mucho más que uno con production=1.

---

## 2. Arquitectura de red

### 2a. Más capas de atención (actual: 1 capa, 2 heads)
**Problema actual:** 1 sola capa de self-attention sobre planetas. Para razonar sobre estrategia global (flanqueo, priorización de targets, defensas coordinadas) hacen falta múltiples capas.

**Mejora:** aumentar a 2–3 capas de atención con residual connections. Coste: ~2x parámetros (sigue siendo pequeño, ~150K).

### 2b. Cross-attention flotas → planetas
**Problema actual:** las flotas se mean-pool en un solo vector `fleet_ctx`. El modelo pierde la información de *qué flota va hacia qué zona*.

**Mejora:** en lugar de mean-pool, usar cross-attention donde cada planeta puede "consultar" las flotas cercanas. Esto le da al modelo conciencia espacial de amenazas locales.

### 2c. Separar action_type del resto del pipeline
**Problema actual:** `action_type_head`, `target_head` y `amount_head` están en paralelo sobre el mismo `h`. El modelo decide target y amount sin saber todavía si va a hacer LAUNCH o NO_OP.

**Mejora (autoregresiva):** primero decide action_type, luego condiciona target y amount sobre esa decisión. Reduce inconsistencias (predecir target cuando la acción es NO_OP no tiene sentido).

### 2d. Tamaño del modelo
**Estado actual:** E=64, F=32, G=128 → 69K params. Pequeño para un juego con dinámica rica.

**Mejora:** probar E=128, F=64, G=256 → ~280K params. En GPU este tamaño es trivial y puede mejorar capacidad expresiva significativamente.

---

## 3. Reward shaping

### 3a. El reward actual es miope
**Problema actual:** `delta_planets + 0.01*delta_ships + 0.1*delta_production`. Todos son deltas de 1 step. El agente puede aprender a ser agresivo a corto plazo aunque pierda a largo.

**Mejora:** añadir un término de **eficiencia**: bonus cuando capturas un planeta gastando menos ships de las que produce en los próximos N turns. Penaliza ataques suicidas.

### 3b. Penalización por inacción tardía
**Problema actual:** no hay penalización por NO_OP cuando estás ganando. El bot puede aprender a "esperar" en posiciones de ventaja en lugar de cerrar la partida.

**Mejora:** en los últimos 100 turns, pequeña penalización por tener ships idle en planetas con vecinos enemigos accesibles.

### 3c. Reward terminal dominante
**Mejora simple y efectiva:** además del shaped reward step-a-step, añadir un **reward terminal grande** (+1 victoria, -1 derrota) al final del episodio. Ancla todo el aprendizaje al objetivo real. El shaped reward entonces solo sirve como señal densa auxiliar, no como objetivo principal.

```python
# Al final del episodio:
terminal_reward = +10.0 if won else -10.0
```

### 3d. Reward relativo, no absoluto
**Problema actual:** `curr_planets - prev_planets` es absoluto. Si el oponente también capturó un planeta ese turn, no deberías recibir reward positivo.

**Mejora:** usar delta relativo:
```python
delta_planets_relative = (my_planets / total_planets) - (prev_my_planets / total_planets)
```

---

## Prioridad recomendada

| Mejora | Impacto | Coste impl. |
|---|---|---|
| Reward terminal (+10/-10) | Alto | Mínimo |
| Reward relativo | Medio-alto | Mínimo |
| `incoming_enemy_ships` en estado | Alto | Medio |
| Más capas de atención (2→3) | Medio | Bajo |
| Tamaño del modelo (E=128) | Medio | Bajo |
| Cross-attention flotas→planetas | Medio | Alto |
| Action type autoregresivo | Medio | Alto |
| Velocidad angular en features | Bajo-Medio | Bajo |
