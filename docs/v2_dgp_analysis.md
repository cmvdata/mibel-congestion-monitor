# v2 — Análisis del DGP del spread ES-FR antes de modelar

**Estado**: borrador para revisión. Sin código predictivo todavía.
**Datos**: `data/processed/mibel_dataset_20190101_20241231.parquet`, $n = 52\,530$ horas, 2019-01-01 a 2024-12-31.
**Numéros**: generados por `scripts/v2_dgp_analysis.py`, dump en `results/v2_dgp_analysis.json`.

---

## 1. Resumen ejecutivo

El spread día-anterior ES-FR no es una serie continua con cola pesada: es **una serie zero-inflated con dos estados físicos discretos**. El 95.55% de las 52,530 horas del periodo 2019-2024 tienen spread **exactamente igual a cero** (interconexión no saturada, arbitraje físico forzando convergencia). El 3.91% restante (2,054 horas) corresponde al estado en que la interconexión está binding y los mercados se desacoplan. La distribución condicional al modo desacoplado tiene mediana 5.9 €/MWh, p95 56 €/MWh y máximo 122 €/MWh.

El modelo de regresión directa sobre `spread_da` con XGBoost+MSE que usa el v1 está estructuralmente mal especificado para este DGP: malgasta la mayor parte de su capacidad prediciendo correctamente que la próxima hora será cero (algo que un modelo trivial M0 también hace). Las features que cuestan datos (renovables, nuclear, demanda) no mejoraron el modelo precisamente porque el modelo no diferencia entre "predecir un cero" y "predecir la magnitud de la cola". **La arquitectura v2 debe ser two-stage hurdle**: un clasificador del estado físico (acoplado vs desacoplado) más un regresor condicional al estado desacoplado.

## 2. Caracterización empírica del DGP

### 2.1 Zero-inflation

| Umbral | n | % |
|---|---:|---:|
| `spread_da == 0` exacto | 50,180 | **95.53%** |
| `\|spread\| ≤ 0.1` | 50,191 | 95.55% |
| `\|spread\| ≤ 0.5` | 50,476 | 96.09% |
| `\|spread\| ≤ 1.0` | 50,567 | 96.27% |
| `\|spread\| ≤ 2.0` | 50,807 | 96.72% |

El salto entre `|s| ≤ 0.5` y `|s| ≤ 0.0` es minúsculo (0.6 puntos porcentuales). El umbral natural de modo es **0.5 €/MWh**: por debajo, mercados acoplados (la diferencia es ruido de tick); por encima, mercados desacoplados.

### 2.2 Persistencia y transiciones

Definimos $S_t = \mathbb{1}\{|spread_t| > 0.5\}$ (binario, 0 = acoplado, 1 = desacoplado).

Matriz de transición empírica:

|  | $S_{t+1} = 0$ | $S_{t+1} = 1$ |
|---|---:|---:|
| $S_t = 0$ | **0.9851** | 0.0149 |
| $S_t = 1$ | 0.3656 | **0.6344** |

Ambos estados son persistentes, pero asimétricamente. La persistencia de "acoplado" es muy alta (98.5%); la de "desacoplado" es moderada (63.4%). El sistema **entra raramente en congestión pero, cuando entra, dura algunas horas antes de relajarse**.

Distribución de run-lengths:

| Estado | n_runs | mean (h) | median (h) | p95 (h) | max (h) |
|---|---:|---:|---:|---:|---:|
| Acoplado | 752 | 67.1 | 23 | 256 | **798** |
| Desacoplado | 751 | **2.7** | 2 | 8 | 17 |

El estado de congestión es típicamente un evento de 2-3 horas. Run máximo: 17h. Esto **define operacionalmente la ventana de detección**: si quieres detectar congestión, tienes que actuar dentro de las 2-3 horas o el evento se acabó.

### 2.3 Distribución condicional al modo desacoplado

Sobre las 2,054 horas con $|spread| > 0.5$:

| Estadístico | Valor |
|---|---:|
| mediana $|s|$ | 5.90 €/MWh |
| p75 $|s|$ | 16.16 |
| p95 $|s|$ | 55.95 |
| p99 $|s|$ | 85.04 |
| max $|s|$ | 122.39 |
| dirección + (ES > FR) | **75.0%** |
| dirección − (FR > ES) | 25.0% |

La cola es brutalmente pesada **incluso dentro del modo desacoplado**, pero ya es una distribución continua con n manejable y estructura modelable. La asimetría direccional 3:1 favorece dirección ES>FR — España tiende a tener precios más altos cuando hay congestión.

### 2.4 Heterogeneidad horaria

Tasa de decoupling por hora del día (datos globales):

- Mínimo: 22h → **0.96%**
- Máximo: 15h → **8.41%**
- Rango: 7.45 puntos porcentuales

La congestión es **8.7× más probable a las 15h que a las 22h**. Esto coincide con peak solar ibérico: cuando España genera mucho solar y Francia no, los precios divergen. Hora del día es una feature de primera línea para Stage A.

### 2.5 Régimen regulatorio

| Régimen | n | exact zero | p99 |s| | max |s| |
|---|---:|---:|---:|---:|
| pre_crisis | 26,253 | 96.04% | 5.97 | 64.87 |
| crisis_y_excepcion | 17,494 | 95.91% | 34.63 | 122.39 |
| post_excepcion | 8,783 | 93.77% | 26.36 | 96.77 |

**Hallazgo importante**: la *frecuencia* de congestión no cambia mucho entre régimenes (93.8%-96.0% acoplado), pero la *magnitud condicional* sí (p99 va de 6 a 35 €/MWh). El régimen no afecta a Stage A pero sí a Stage B.

## 3. Definición operacional de modos

$$
S_t = \begin{cases} 0 \text{ (acoplado)} & \text{si } |spread_t| \leq 0.5 \text{ €/MWh} \\ 1 \text{ (desacoplado)} & \text{si } |spread_t| > 0.5 \text{ €/MWh} \end{cases}
$$

El umbral de 0.5 €/MWh está justificado por la concentración de masa en `|s| ≤ 0.5` (96.09%) frente a `|s| = 0` (95.53%). El intervalo (0, 0.5] €/MWh contiene 296 horas (0.56%) que probablemente son ruido de tick o redondeo del clearing — no congestión real. Por encima de 0.5 ya hay separación física entre los mercados.

## 4. Arquitectura propuesta

### Stage A: Clasificador de estado

$$
\hat{P}(S_{t+1} = 1 \mid \mathbf{x}_t)
$$

- **Modelo candidato base**: logistic regression con interacciones explícitas.
- **Modelo candidato avanzado**: gradient boosted classifier (LightGBM con `objective='binary'`).
- **Métrica primaria**: AUC-ROC y AUC-PR (importante con clase fuertemente desbalanceada).
- **Métrica operacional**: lead time mediano a la transición $S_t: 0 \to 1$.

### Stage B: Regresión condicional al estado desacoplado

$$
\hat{E}[spread_{t+1} \mid S_{t+1} = 1, \mathbf{x}_t]
$$

Solo se entrena sobre las 2,054 horas con $S = 1$. Como la cola sigue siendo pesada dentro del modo, mejor especificación: quantile regression a $q = 0.5$ + $q = 0.95$ para tener intervalos de predicción condicionales.

### Predicción combinada

$$
\hat{spread}_{t+1} = \hat{P}(S_{t+1} = 1) \cdot \hat{E}[spread \mid S_{t+1} = 1]
$$

## 5. Features propuestas con justificación física

### Para Stage A (clasificar congestión, target binario)

| Feature | Justificación física |
|---|---|
| Hora del día (seno/coseno) | 8.7× más decoupling 15h vs 22h |
| Día de la semana | Demanda industrial diferencial ES-FR fin de semana |
| Mes / estación | Solar peak ibérico vs demanda invernal francesa |
| Festivos ES, festivos FR | Asimetría de demanda en festivos no comunes |
| `ntc_es_fr`, `ntc_fr_es` actuales | Capacidad disponible — limitante físico directo |
| `ntc_is_observed` flag | Diferenciar interpolado pre-JAO del real post-2022 |
| Lag 24h del estado $S_{t-24}$ | "¿Estaba congestionado ayer a esta hora?" — patrones diarios |
| Lag 168h del estado $S_{t-168}$ | Patrones semanales |
| Run-length actual del estado | Si llevo 5h en modo congestionado, P(continuar) > P(salir) |

Lo que NO usamos en Stage A: TTF, CO2, lags del spread continuo. Esas señales son irrelevantes para clasificar binario (la congestión depende de capacidad física, no de coste marginal).

### Para Stage B (regresión condicional)

Entran las features que TTF, CO2 y diferenciales aportan, porque ahora SÍ definen la magnitud:

| Feature | Justificación |
|---|---|
| `ttf_eur_mwh`, `co2_eur_t`, `spark_spread` | Set the absolute price floor |
| `ntc_es_fr` (severidad de saturación, no binario) | Más restricción → más spread |
| Diferencial renovable ES−FR (si conseguimos datos físicos) | Causa estructural de asimetría |
| Diferencial demanda ES−FR | Idem |
| Lag de magnitud previa $|spread_{t-1}|$ condicional a $S_{t-1}=1$ | Persistencia dentro del modo |
| Régimen regulatorio (dummy) | Magnitud condicional varía mucho entre régimenes (p99 5.97 → 34.63) |

## 6. Esquema de validación

- **Walk-forward semanal**: reentrenar Stage A y Stage B cada semana sobre todos los datos hasta $t-1$, evaluar sobre la semana siguiente. No régime-split fijo.
- **Métricas Stage A**: AUC-ROC, AUC-PR, Brier score, calibración (reliability plot), lead time a transición.
- **Métricas Stage B**: RMSE y pinball loss q=0.95 condicionales al estado desacoplado real ($S = 1$).
- **Métrica end-to-end**: lead time global (cuándo se dispara la alerta antes de la próxima transición $0 \to 1$) a un FPR fijo (10 alertas/mes operativamente razonable).

## 7. Hipótesis falsable

> **H1**: un clasificador logístico con las features físicas listadas para Stage A alcanza AUC-ROC > 0.95 y AUC-PR > 0.60 en validación walk-forward semanal sobre 2024.

Si H1 se confirma → la arquitectura two-stage está justificada y avanzamos a Stage B con tranquilidad.
Si H1 falla (AUC-ROC < 0.85 o AUC-PR < 0.30) → el DGP es más complejo que dos estados, hay que considerar hidden Markov de 3+ estados o régimen-switching estructural.

**Métrica de éxito para Stage B condicional**: pinball loss q=0.95 dentro del modo desacoplado < 0.5 × pinball loss del v4-quantile actual aplicado al mismo subset. Es decir, el regresor condicional debe ser al menos el doble de preciso que el modelo no condicional dentro de la región de interés.

## 8. Próximos pasos

1. **Revisión** de este documento. Si hay objeciones al threshold de 0.5 €/MWh, al esquema de validación o a la hipótesis falsable, las discutimos antes de tocar código.
2. **Implementación de Stage A** (clasificador) — 2-3 días si las features están disponibles.
3. **Test de H1**: walk-forward semanal sobre 2024, reporte de AUC y lead time.
4. **Decisión sobre Stage B** condicional a si H1 se confirma.
5. **Datos pendientes** que harían Stage A más rico: flujos físicos REE/RTE (no solo NTC), festivos por país, forecasts intraday OMIE.

## 9. Lo que esto NO es

- **No es** un cambio de modelo dentro de la arquitectura v1. Es una arquitectura distinta.
- **No es** una mejora marginal del XGBoost. Es reconocer que el DGP exige una estructura distinta.
- **No es** garantía de que la nueva arquitectura mejore las métricas operativas. Es la arquitectura que el DGP demanda; la validación empírica de H1 dirá si funciona.
- **No es** publicable todavía. Es la fase previa para no repetir el v1.
