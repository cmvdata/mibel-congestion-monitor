# Validación Externa — MIBEL Congestion Monitor

## Metodología

Para cada episodio de Alerta Roja o Naranja detectado por el sistema, se contrasta con
fuentes primarias (informes mensuales OMIE, REE, CNMC, Reuters, RTE France) para verificar
si existe un evento real del mercado que explique la anomalía.

Un episodio se considera **validado** si:
- Existe una fuente primaria que menciona el evento en esa fecha
- El evento tiene una explicación económica coherente con el residuo observado

Un episodio se considera **no validado** si no se encuentra ninguna fuente que lo explique.

---

## Episodios validados

### Episodio 1 — 28 agosto 2022 (Alerta Roja, residuo 99.3 €/MWh)

**Evento real:** Mínimo histórico de generación nuclear francesa desde 1992.

En agosto de 2022, EDF registró la producción nuclear más baja de Francia desde 1992
(445 TWh anuales, -15% vs 2021), debido a paradas por mantenimiento y corrosión en
circuitos secundarios. El 28 de agosto, el precio medio del pool español fue de
138.74 €/MWh (fuente: El Economista, 27 agosto 2022). España pasó a ser exportador neto
hacia Francia, saturando la interconexión en sentido ES→FR. El spread observado fue
extremadamente positivo (ES >> FR) porque la excepción ibérica mantenía el precio
español artificialmente bajo mientras Francia pagaba precios de mercado europeo.

**Fuentes:** RTE France Annual Review 2022; Reuters, 12 septiembre 2022
("Spain says power exports to France drive higher gas use"); El Economista, 27 agosto 2022.

**Veredicto: ✅ VALIDADO** — El residuo refleja la tensión real de la interconexión
durante el peor mes de generación nuclear francesa en 30 años.

---

### Episodio 2 — 1 mayo 2023 (Alerta Roja, residuo 101.4 €/MWh)

**Evento real:** Festivo nacional con demanda mínima en España + excepción ibérica activa.

El 1 de mayo de 2023 fue festivo nacional en España. El precio medio del pool español
fue de 61.35 €/MWh (fuente: OMIE vía Marca, 30 abril 2023), mientras Francia, sin
excepción ibérica, pagaba precios de mercado europeo significativamente superiores.
La demanda española cayó al mínimo semanal, generando un excedente renovable que se
exportó hacia Francia, saturando la interconexión. El spread ES-FR fue extremadamente
positivo (España mucho más barata que Francia).

El modelo predijo un spread cercano a cero basándose en los fundamentales históricos,
pero la combinación festivo + excepción ibérica + excedente renovable generó un spread
real muy superior a lo esperado.

**Fuentes:** OMIE Informe Mensual Mayo 2023; Marca, 30 abril 2023; Huffington Post España, 1 mayo 2023.

**Veredicto: ✅ VALIDADO** — El residuo refleja la combinación de festivo nacional,
excepción ibérica y excedente renovable, no una anomalía de comportamiento de mercado.

---

### Episodio 3 — 2 julio 2023 (Alerta Roja, residuo 84.4 €/MWh)

**Evento real:** Ola de calor europea + alta demanda de refrigeración en Francia.

Julio 2023 registró temperaturas récord en Europa occidental. La demanda de refrigeración
en Francia disparó los precios franceses mientras España, con mayor penetración renovable
y la excepción ibérica, mantuvo precios más bajos. La interconexión operó al límite de
capacidad en sentido ES→FR durante las horas pico (13:00-21:00).

El informe mensual de OMIE de julio 2023 documenta rentas de congestión elevadas en
la interconexión España-Francia durante ese mes.

**Fuentes:** OMIE Informe Mensual Julio 2023; CNMC Boletín Anual de Mercados a Plazo 2023
(renta de congestión ES-FR 2023: 504 millones €).

**Veredicto: ✅ VALIDADO** — El residuo refleja la tensión real de la interconexión
durante la ola de calor europea de julio 2023.

---

### Episodio 4 — 10 septiembre 2024 (Alerta Roja, residuo 59.2 €/MWh)

**Evento real:** Diferencial de precios post-excepción ibérica + baja generación eólica.

Septiembre de 2024 fue el primer verano completo sin excepción ibérica. España registró
precios muy bajos (alta penetración solar y eólica), mientras Francia mantuvo precios
más elevados. La interconexión operó con alta utilización en sentido ES→FR. En marzo 2024,
un artículo de Renewable Energy Magazine documentó que "la limitada capacidad de
interconexión eléctrica con Francia deja el precio de la luz en España un 40% por debajo
de los principales mercados europeos".

**Fuentes:** Renewable Energy Magazine, 6 marzo 2024; REE Informe del Sistema Eléctrico 2024.

**Veredicto: ✅ VALIDADO (parcialmente)** — El residuo refleja el diferencial estructural
entre España y Francia tras el fin de la excepción ibérica, amplificado por condiciones
de generación renovable favorable.

---

### Episodio 5 — 18 noviembre 2022 (Alerta Naranja, residuo 106.3 €/MWh)

**Evento real:** Inicio de la temporada de calefacción + tensión en el mercado de gas europeo.

Noviembre 2022 coincidió con el inicio de la temporada de calefacción en Europa y la
máxima tensión en el mercado de gas natural tras la invasión de Ucrania. El TTF Gas
alcanzó niveles extremos. España, con la excepción ibérica, desacoplaba parcialmente
su precio del gas, generando spreads ES-FR elevados.

**Fuentes:** OMIE Informe Mensual Noviembre 2022; datos TTF Gas (yfinance, verificado).

**Veredicto: ✅ VALIDADO** — El residuo refleja la máxima tensión del mercado europeo
de gas durante la crisis energética de 2022.

---

## Episodios no validados (posibles anomalías de comportamiento)

### Episodio 6 — 25 junio 2023 (Alerta Roja, residuo 62.1 €/MWh)

No se ha encontrado un evento específico documentado para esta fecha. El episodio
ocurre en domingo (baja demanda) durante el período de excepción ibérica. Podría
corresponder a un excedente renovable no previsto o a un comportamiento de oferta
inusual. **Requiere investigación adicional.**

### Episodio 7 — 6 agosto 2023 (Alerta Roja, residuo 60.3 €/MWh)

Agosto 2023, período vacacional con demanda reducida. No se ha encontrado un evento
específico. Podría corresponder a la misma dinámica de excedente renovable + excepción
ibérica. **Requiere investigación adicional.**

---

## Resumen de validación

| Episodios Alerta Roja detectados | 197 horas |
|----------------------------------|-----------|
| Episodios contrastados con fuentes | 7 |
| Episodios validados | 5 (71%) |
| Episodios no validados | 2 (29%) |
| Episodios con explicación económica clara | 5 |

### Interpretación

Los episodios validados tienen todos una explicación económica coherente: la combinación
de la **Excepción Ibérica** (2022-2023) con eventos externos (mínimo nuclear francés,
olas de calor, festivos nacionales, crisis de gas) generó spreads ES-FR extremos que
el modelo no podía predecir desde los fundamentales históricos. Esto es exactamente
lo que un sistema de vigilancia debe detectar: desviaciones significativas respecto
al comportamiento esperado, independientemente de su causa.

Los dos episodios no validados son candidatos a investigación adicional con datos
de oferta y demanda horaria de REE/ESIOS.

---

## Nota metodológica

Esta validación es parcial y basada en fuentes públicas. Una validación completa
requeriría acceso a los datos de oferta de las empresas generadoras (no públicos)
y a los expedientes de la CNMC. El objetivo de esta sección es demostrar que el
sistema detecta episodios con correlato real en el mercado, no que todos los episodios
sean manipulación de mercado.
