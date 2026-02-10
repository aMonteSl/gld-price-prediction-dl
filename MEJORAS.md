# MEJORAS.md — Análisis Estratégico y Propuesta de Arquitectura UX

**Documento de Planificación Estratégica**  
**Fecha:** Febrero 2026  
**Alcance:** Realineación de la aplicación hacia la pregunta central del usuario

---

## 1. Resumen Ejecutivo

### 1.1 ¿Qué es la aplicación hoy?

Una plataforma técnicamente rica, impulsada por investigación, que:
- Descarga datos históricos de múltiples activos financieros
- Entrena modelos de deep learning (TCN, LSTM, GRU) con pérdida cuantil
- Genera pronósticos probabilísticos (P10/P50/P90)
- Produce planes de acción diarios con recomendaciones BUY/HOLD/SELL/AVOID
- Compara portafolios y calcula métricas de riesgo

**Estructura actual:** 8 pestañas independientes (Data › Train › Models › Forecast › Recommendation › Evaluation › Compare › Tutorial)

**Público implícito:** Traders técnicos, investigadores en fianzas cuantitativas, educadores.

### 1.2 ¿Qué debería ser?

Una herramienta de **toma de decisión enfocada en inversiones**, donde:
- El usuario articuló una pregunta clara: *"Tengo X dinero. ¿Debería invertir en este activo hoy o esperar? Si invierto, ¿cuándo vendo?"*
- La respuesta llega **en menos de 30 segundos** sin requerir entrenamiento de modelos
- Los detalles técnicos están disponibles pero **no obligan** al flujo principal
- La confianza en la recomendación se construye a través de **visualización de riesgo, escenarios y rationales claros**
- Los modelos pre-entrenados son **activos de larga vida**, reutilizables y comparables

**Público objetivo:** Inversores independientes, pequeños fondos, educandos en finanzas cuantitativas, traders interesados en análisis técnico profundo.

### 1.3 Problemas principales

| Problema | Impacto | Nivel |
|----------|--------|-------|
| 8 pestañas independientes sin jerarquía clara | El usuario no sabe por dónde empezar. ¿Necesito entrenar un modelo? | 🔴 Crítico |
| El entrenamiento es obligatorio en el flujo primario | Fricción. Mayoría no quiere entrenar; quieren analizar con modelos existentes | 🔴 Crítico |
| Recomendación está "al final" del tubo, no en el centro | La pregunta clave ("¿debo invertir?") no es el destino sino una parada | 🔴 Crítico |
| Selección de modelos es opaca | ¿Cuál modelo elegir? ¿Por qué uno sobre otro? No hay guía | 🟡 Alto |
| Pronóstico y Recomendación están desacoplados | El usuario no ve cómo los escenarios (P10/P50/P90) generan decisiones | 🟡 Alto |
| Compare requiere modelos pre-asignados a activos | Workflow no intuitivo; requiere pasos previos en Train | 🟡 Alto |
| La incertidumbre (P10/P50/P90) no es visualmente clara | El riesgo de pérdida no es inmediatamente comprensible | 🟡 Alto |
| No hay "vista ejecutiva" rápida | Inversores rápidos no pueden explorar múltiples activos en segundos | 🟡 Alto |
| Técnica domina la experiencia | Un usuario casual se ahoga en Loss Curves, RSI, Feature Engineering | 🟡 Alto |
| Disclaimer/Educación débil | La app se presenta como neutra; pero hay recomendaciones | 🔴 Crítico |

---

## 2. Arquitectura de Alto Nivel Propuesta

### 2.1 Concepto: Dos Modos de Operación

```
┌─────────────────────────────────────────────────────────┐
│  🎯 MODO INVERSOR (Decision-Primary)                    │
│  "¿Debería invertir?"  →  Mostrar respuesta clara      │
│  Usuarios: Inversores independientes, traders           │
└─────────────────────────────────────────────────────────┘
                         │
                    Opcional: Vista Técnica
                         │
┌─────────────────────────────────────────────────────────┐
│  🔬 MODO INVESTIGADOR (Analysis-Primary)                │
│  "¿Qué hace el modelo?"  →  Mostrar todos los detalles │
│  Usuarios: Quants, educadores, ingenieros              │
└─────────────────────────────────────────────────────────┘
```

### 2.2 Estructura Propuesta de Pestañas (Reorganizada)

**En lugar de 8 pestañas planas:**

#### **Nivel 1: Flujo Primario (Decision)**

```
┌──────────────────────────────────────────────────────────┐
│ 1. 📊 DASHBOARD (Nueva)                                  │
│    - Entrada principal                                   │
│    - Selector de activo                                 │
│    - Última recomendación (BUY/HOLD/SELL/AVOID)       │
│    - Resumen de escenarios (P10/P50/P90 en $$)        │
│    - Leaderboard de múltiples activos                  │
│    - "Hoy: ¿Debo entrar o esperar?"                   │
└──────────────────────────────────────────────────────────┘
         ↓
┌──────────────────────────────────────────────────────────┐
│ 2. 🎯 RECOMENDACIÓN (Expandida)                         │
│    - Plan de acción detallado (BUY/HOLD/SELL/AVOID)   │
│    - Timeline color-codificado                          │
│    - Escenarios con P&L en dinero                      │
│    - Rationales de decisión (4 factores)              │
│    - Ventana de entrada / punto de salida             │
│    - Advertencias, métricas de confianza              │
│    - Gráfico interactivo activo                       │
└──────────────────────────────────────────────────────────┘
         ↓
┌──────────────────────────────────────────────────────────┐
│ 3. ⚖️ COMPARAR (Rediseñado)                             │
│    - Vista tipo scorecard de múltiples activos         │
│    - Asignación rápida de monto de inversión          │
│    - Ranking por median return / Sharpe / ratio riesgo │
│    - Gestión de modelos (asignar a activos)          │
└──────────────────────────────────────────────────────────┘
```

#### **Nivel 2: Flujo Técnico (Investigación)**

```
┌──────────────────────────────────────────────────────────┐
│ 4. 🔍 ANÁLISIS FORECAST (Fue la pestaña Forecast)      │
│    - Fan charts (P10/P50/P90)                          │
│    - Visualización de incertidumbre por día           │
│    - Linkaje explícito a decisiones BUY/SELL         │
│    - Inspección de probabilidades                      │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│ 5. 📈 EVALUACIÓN (Fue la pestaña Evaluation)            │
│    - Métricas de trayectoria (MSE/RMSE/MAE)           │
│    - Calibración de cuantiles                          │
│    - Análisis de desempeño histórico                   │
│    - Backtesting simulado (opcional: nuevo)           │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│ 6. 🏋️ MODELOS (Expandida)                               │
│    - Galería de modelos guardados                      │
│    - Selección de modelo primario por activo           │
│    - Comparación por arquitectura (TCN vs LSTM)       │
│    - Inspección de metadata / fecha de entrenamiento  │
│    - Eliminar / renombrar modelos                     │
│    - Sugerir modelo basado en performance             │
└──────────────────────────────────────────────────────────┘
```

#### **Nivel 3: Entrenamiento (Opcional, Avanzado)**

```
┌──────────────────────────────────────────────────────────┐
│ 7. 🚂 ENTRENAR MODELO (Redefinido como "Avanzado")      │
│    - Buscador de datos (selecciona activo/período)    │
│    - Configuración de arquitectura (TCN/LSTM/GRU)     │
│    - Hiperparámetros de entrenamiento                  │
│    - Visualización en vivo de pérdida                  │
│    - Diagnósticos automáticos con sugerencias         │
│    - Fine-tuning desde modelo existente               │
│    - Nombrado personalizado del modelo                 │
│    → Envío automático al registry como "candidato"    │
│    → Comparación vs modelo actual (por métrica)       │
└──────────────────────────────────────────────────────────┘
```

#### **Nivel 4: Educación**

```
┌──────────────────────────────────────────────────────────┐
│ 8. 📚 TUTORIAL & GLOSARIO (Rediseñado)                  │
│    - Guía interactiva paso a paso                      │
│    - Explainadores en contexto (hover → definiciones) │
│    - Videos conceptuales (opcional)                    │
│    - FAQ sobre interpretación de recomendaciones      │
│    - Advertencias sobre límites del modelo            │
│    - Escenarios de ejemplo con narrativa              │
└──────────────────────────────────────────────────────────┘
```

### 2.3 Flujo de Redirección

```
Usuario nuevo accede a la app
         ↓
    [DASHBOARD]  ← Entrada primaria
         ↓
   ¿Qué debo hacer?
    /      |      \
  SÍ     ESPERAR   NO
   ↓        ↓       ↓
[RECO]  [Esperar] [RECO]
   ↓              ↓
¿Cómo ejecuto?  [Dashboard en N días]
   ↓
[Educ+Warnings]

Flujo Técnico (paralelo, siempre disponible):
[ANÁLISIS] → [EVAL] → [MODELOS] → [ENTRENAR]
```

---

## 3. Flujo de Usuario Mejorado

### 3.1 "Happy Path" — El Inversor Casual (30 segundos)

```
1. Abre la app → Aparece DASHBOARD
2. Lee: "GLD hoy: HOLD, confianza 78%, downside -2.1%, upside +4.8%"
3. Ve timeline: "Entrar en 3-5 días, salir en 12"
4. Presiona "¿Por qué?" → Popover con rationale de 4 factores
5. Presiona "Escenarios" → P10/P50/P90 en dinero:
   - Pesimista: Perder $200
   - Base: Ganar $450
   - Optimista: Ganar $950
6. Presiona "COMPARAR CON SLV" → Leaderboard muestra SLV y GLD lado a lado
7. Decide: "Prefiero GLD" → Confirma y archiva decisión
```

**Tiempo total: ~30 segundos**  
**Acciones técnicas requeridas: 0**

---

### 3.2 "Intelligent Path" — El Investigador (5–10 minutos)

```
1. Abre DASHBOARD → Selecciona activo
2. Hace clic en "ANÁLISIS FORECAST"
   - Ve fan chart detallado con bandas P10/P50/P90
   - Hace hover a días específicos para ver probabilidades
3. Pregunta: "¿Por qué el modelo prefiere esperar?"
   - Va a EVALUACIÓN
   - Inspecciona métricas de calibración de cuantiles
   - Revisa desempeño histórico en volatilidad alta
4. Pregunta: "¿Qué modelo se está usando?"
   - Va a MODELOS
   - Ve que está usando "TCN_GLD_performance_v2" entrenado el 1 feb
   - Compara vs otros modelos disponibles (LSTM, GRU variantes)
5. Considera: "¿Puedo entrenar algo mejor?"
   - Va a ENTRENAR MODELO
   - Ajusta hiperparámetros
   - Corre entrenamiento en vivo (observa loss curve)
6. Nuevo modelo termina:
   - Sistema sugiere: "Mejor RMSE que v2, pero P50 menos conservador"
   - Compara recomendación del nuevo modelo vs actual
   - Puede asignar nuevo modelo como primario o guardar como "experimento"
7. Vuelve a RECOMENDACIÓN
   - Ve cómo la recomendación cambió (o no) con nuevo modelo
```

**Tiempo total: 8–12 minutos**  
**Acción técnica requerida: Sí, pero informada**

---

### 3.3 "Comparison Path" — El Gestor de Portafolio (10 minutos)

```
1. Abre DASHBOARD
2. Lee resumen de 4 activos (GLD, SLV, BTC, PALL)
3. Presiona "⚖️ COMPARAR"
4. Ingresa monto: $10,000
5. Sistema ejecuta:
   - Descarga datos más recientes (cache 1h)
   - Carga modelos primarios asignados a cada activo
   - Genera escenarios P10/P50/P90 para horizonte de 20 días
   - Calcula Sharpe, ratio riesgo/recompensa, max drawdown
6. Sistema muestra leaderboard (ordenado por median return):
   |  # | Activo | Modelo | P50 Return | Max Loss | Convicción |
   | -- | ------ | ------ | ---------- | -------- | ---------- |
   | 1  | BTC    | LSTM_v3| +8.2%      | -3.1%    | 72% BUY    |
   | 2  | GLD    | TCN_v2 | +2.1%      | -1.8%    | HOLD       |
   | 3  | SLV    | LSTM_v3| +1.5%      | -2.4%    | HOLD       |
   | 4  | PALL   | GRU_v1 | -0.5%      | -3.2%    | AVOID      |
7. Hace clic en BTC → Expande y ve:
   - Plan detallado (cuándo entrar/salir)
   - Rationale de decisión
   - P10/P50/P90 con distribución de probabilidad
8. Asigna dinamámicamente: $6k a BTC, $3k a GLD
9. Archiva "portafolio del 10 de febrero" para revisión futura
```

**Tiempo total: 10–15 minutos**  
**Acciones técnicas: Selecciones, sin entrenamiento**

---

## 4. Rediseño del Sistema de Recomendación

### 4.1 Información que DEBE mostrar (Obligatorio)

```
┌─────────────────────────────────────────────────────────┐
│ ACCIÓN RECOMENDADA HOYMENTE (Visible de inmediato)     │
│ ┌───────────────────────────────────────────────────────┤
│ │ 🟢 BUY  |  Confianza 76%                              │
│ │ "Horizonte de 15 días, ventana de entrada 3–5"        │
│ └───────────────────────────────────────────────────────┘
│ NARRATIVE (Una oración, lenguaje natural)               │
│ "SMA-50 cruzó SMA-200 alcista. ATR en régimen normal.  │
│  P10 drawdown dentro de tolerancia. Hoy es óptimo       │
│  para entrar; considere acumular en los próximos 3–5.'" │
└─────────────────────────────────────────────────────────┘
```

### 4.2 Información que DEBE mostrarse (Al expandir)

```
┌─────────────────────────────────────────────────────────┐
│ RATIONALE DE 4 FACTORES (Expandible)                    │
│ ┌───────────────────────────────────────────────────────┤
│ │ 1️⃣  Confirmación de Tendencia                         │
│ │     SMA-50 > SMA-200 ✓  (Golden cross activo)        │
│ │     Señal: ALCISTA                                    │
│ │                                                        │
│ │ 2️⃣  Régimen de Volatilidad                           │
│ │     ATR% = 1.8%  →  NORMAL (histórico: 1.2–2.4%)   │
│ │     Interpretación: No es ni muy calmo ni muy turbio │
│ │                                                        │
│ │ 3️⃣  Riesgo de Cuantil                               │
│ │     P10 (worst case): -2.1% en horizonte de 20 días │
│ │     Tu SL está en -3.0% → Margen: +0.9%             │
│ │     Estado: SEGURO (pérdida máxima dentro de límite) │
│ │                                                        │
│ │ 4️⃣  Evaluación del Día de Hoy                        │
│ │     Puntuación técnica: 0.72 / 1.0                   │
│ │     ¿Es hoy óptimo? SÍ (dentro de ventana de entrada)│
│ └───────────────────────────────────────────────────────┘
```

### 4.3 Escenarios con Impacto en Dinero

```
┌───────────────────────────────────────────────────────────┐
│ TRES ESCENARIOS DE PRECIOS (20 días)                      │
│ Tu inversión inicial: $10,000  |  Precio de entrada: $195 │
├───────────────────────────────────────────────────────────┤
│ 🔴 PESIMISTA (P10, 10% de probabilidad)                  │
│    Precio final: $190.54  →  Cambio: -2.3%              │
│    P&L: -$230   |  Precio máximo alcanzado: $192        │
│                                                            │
│ 🟡 BASE (P50, mediana)                                    │
│    Precio final: $199.21  →  Cambio: +2.2%              │
│    P&L: +$220   |  Max DD (drawdown): -1.1%              │
│    Días al máximo: 12 (puedes salir antes)               │
│                                                            │
│ 🟢 OPTIMISTA (P90, 10% de probabilidad)                  │
│    Precio final: $207.35  →  Cambio: +6.4%              │
│    P&L: +$640   |  Precio máximo alcanzado: $209        │
│    Este es el mejor escenario...                         │
│    ...pero es raro (1 de 10 veces)                      │
├───────────────────────────────────────────────────────────┤
│ ⚡ RESUMEN DE RIESGO                                      │
│ Resultado esperado (mediana): +$220 (+2.2%)             │
│ Rango probable: -$230 a +$640                            │
│ Ratio riesgo/recompensa: 1:2.8  (bueno)                 │
│ Máxima pérdida posible: -$500 (5%)                      │
│ Confianza en P50: 72%                                    │
└───────────────────────────────────────────────────────────┘
```

### 4.4 Timeline Interactiva de Acciones

```
Día 1  |  Día 2  |  Día 3  |  Día 4  |  Día 5  | Día 6–20
  ✓    |   ✓     |   ✓     |  HOLD   |  HOLD   |  (varía)
 BUY   |  BUY    |  BUY    | Dentro  | Dentro  |
 ¿Hoy? | Seguir  | Seguir  | Posición| Posición| SELL en
       | entrando| entrando|         |         | día 14
                                              ↓
                                         Salir acá
                                      +2.1% (mediana)
                   ---Expandible---
                   Haz clic en un día
                   para más detalles
           ┌───────────────────────┐
           │ Día 3: BUY            │
           │ P50: $198.20          │
           │ Confianza: 0.74       │
           │ Razón: ATR estable    │
           │ Mejor aún que día 2   │
           └───────────────────────┘
```

### 4.5 Gestión de Confianza y Warnings

```
┌─────────────────────────────────────────────────────────┐
│ 🔔 WARNINGS & CONTEXTO                                  │
├─────────────────────────────────────────────────────────┤
│ ⚠️  Este modelo fue entrenado hace 8 días.              │
│    Recalibración recomendada en 3 días.                 │
│                                                          │
│ 💡  La volatilidad está en máximos de 6 meses.        │
│    Aumenta el riesgo. Considera reduce posición.       │
│                                                          │
│ ℹ️  Modo "educación" activado. Las recomendaciones    │
│    son hipotéticas y solo para análisis.               │
│    No es un asesor financiero.                         │
│                                                          │
│ 📊  Última actualización: hoy a las 16:45              │
│    Datos de mercado: ~30 min de retraso                │
│                                                          │
│ 🎯  Precisión histórica:                               │
│    Últimas 30 recs: 52% ganador, 48% perdedor         │
│    Sharpe ratio: 0.58 (modesto)                        │
│    Max drawdown nunca excedió SL por >1.2%            │
└─────────────────────────────────────────────────────────┘
```

---

## 5. Mejoras en Pronóstico y Riesgo

### 5.1 Visualización de Fan Chart Mejorada

**Cambio conceptual:** El fan chart NO es solo visualización bonita.  
Es el **puente entre el pronóstico y la decisión**.

```
Current state:
  Fan chart con bandas P10/P50/P90
  El usuario ve líneas coloridas
  ¿Qué hacer con ellas? → Vago

Proposed state:
  Fan chart + Shading de decisiones
  - Zona VERDE: "óptimo entrar aquí"
  - Zona AMARILLA: "mantener posición"
  - Zona ROJA: "salir, stop-loss tocado"
  - Línea NEGRA: "salida óptima (máx Sharpe)"
  
  + Anotaciones de texto:
    - "Entrada en día 3–5" (cuando P50 está en verde)
    - "Salida en día 12" (when risk-adj return is peak)
    - "Riesgo si esperas: P10 toca -3.0% en día 18"
```

### 5.2 Métricas de Incertidumbre Explícitas

Para cada día del horizonte:

```
Día  |  P50    |  P10–P50  |  P50–P90  |  Ancho Total  |  Confianza
-----|---------|-----------|-----------|---------------|----------
1    |  +0.2%  |  -1.8%    |  +1.5%    |  3.3%         | 74%
2    |  +0.5%  |  -1.9%    |  +1.8%    |  3.7%         | 72%
3    |  +1.1%  |  -1.7%    |  +2.1%    |  3.8%         | 71% ✓ BUY
4    |  +1.8%  |  -1.5%    |  +2.5%    |  4.0%         | 70%
5    |  +2.2%  |  -1.3%    |  +3.2%    |  4.5%         | 68%
...  |  ...    |  ...      |  ...      |  ...          | ...
20   |  +4.1%  |  -2.1%    |  +5.8%    |  7.9%         | 48% ← uncertain

Interpretación:
- Días 1–5: "Modelo está seguro de la dirección"
- Día 20: "Mucha incertidumbre, no recomendable como horizonte"
```

### 5.3 Escenarios Vinculados a Decisiones

```
P10 (Pesimista)
│  → ¿Alcanza tu SL?
│     SÍ  →  Potencial pérdida grande, AVOID o reduce
│     NO  →  Tolerable, mantén plan
│
P50 (Base)
│  → ¿Supera tu TP o min return?
│     SÍ  →  SELL en TP
│     NO  →  HOLD, espera
│
P90 (Optimista)
│  → Mejor caso. No sobre-confiar.
│     Editorial: "1 de 10 veces"
```

---

## 6. Estrategia de Gestión de Modelos

### 6.1 Ciclo de Vida de un Modelo

```
VERSIÓN              ESTADO          ACCIÓN TÍPICA
───────────────────────────────────────────────────
GLD_TCN_v1  ────→  "Actual"         Usado para decisiones
                   (Asignado)       Comparado vs otros
                                    Recalibrado Después de N días

GLD_TCN_v2  ────→  "Candidato"      Recién entrenado
                   (No asignado)    Comparación vs actual
                                    A/B testing (opcional)

GLD_LSTM_v1 ────→  "Archivo"        Modelos viejos
                   (Histórico)      Mantener para Backtesting
                                    Análisis post-mortem

Nuevo modelo:
  1. Usuario entrena en "ENTRENAR MODELO"
  2. Sistema sugiere: "Mejor RMSE (16.2 vs 18.3)"
  3. Usuario elige:
     a) Promover a "Actual" → Cambia recomendación
     b) Comparar lado-a-lado → Inspecciona A/B
     c) Archivar → Guarda para historia
```

### 6.2 Recomendación de Modelo

```
┌─────────────────────────────────────────────────────────┐
│ 💡 SUGERENCIA DE SISTEMA                                │
│                                                          │
│ "Para GLD, detecté 3 modelos candidatos:"              │
│                                                          │
│ 1️⃣  GLD_TCN_v2  (Actual)                              │
│     RMSE: 18.3  |  Sharpe histórico: 0.62              │
│     Entrenado: 8 días ago                              │
│     Calibración: Excelente en volatilidad normal       │
│     ✓ RECOMENDADO (confiable)                         │
│                                                          │
│ 2️⃣  GLD_LSTM_v3 (Experimento nuevo)                   │
│     RMSE: 16.2  |  Sharpe histórico: 0.58              │
│     Entrenado: 2 horas ago                             │
│     ⚠️  Backtesting limitado (pocos datos)             │
│     → Test 5 más días antes de asignar                │
│                                                          │
│ 3️⃣  GLD_GRU_v1  (Viejo)                               │
│     RMSE: 24.1  |  Sharpe histórico: 0.41              │
│     Entrenado: 6 meses ago                             │
│     ❌ NO RECOMENDADO (desactualizado)                 │
│     → Considere reentrenar si quiere GRU              │
│                                                          │
│ [Asignar TCN_v2] [Probar LSTM_v3]  [Opciones...]      │
└─────────────────────────────────────────────────────────┘
```

### 6.3 Prevención de "Reentrenamiento Innecesario"

Problema actual: El usuario entrena obsesivamente buscando "el mejor modelo".

Solución:

```
1. Mostrar histórico de desempeño:
   "Última 20 modelos entrenados para GLD.
    Mejora median: +0.8% en Sharpe.
    Diminishing returns después de v15."

2. Sugerir pausas:
   "Tu modelo actual tiene 5 días.
    Calibración aún excelente.
    No entrenes hasta que loss curve se degrade."

3. Validar mejora:
   "Nuevo modelo RMSE es 2.1% mejor.
    Pero Sharpe histórico es PEOR (-0.05).
    ¿Seguro que quieres asignarlo?"

4. Backtesting automático:
   "LSTM_v3 tenía -8% más drawdown en 2024.
    ¿Quieres ejecutar anyway?"
```

---

## 7. Sugerencias de UX e Interfaz

### 7.1 Paleta de Colores de Acciones

```
Recomendación   Color     Hexadecimal    Significado
──────────────────────────────────────────────────
BUY             Verde     #27ae60 ✓     Entrar ahora
HOLD            Naranja   #f39c12 ⏸    Mantener, no actuar
SELL            Rojo      #e74c3c ✗    Salir / no entrar
AVOID           Gris      #7f8c8d ⊘    Evitar completamente

Confianza
──────────────────────────────────────────────────
Alta (75%+)     Verde oscuro  #1e8449
Media (50–74%)  Verde claro   #52be80
Baja (<50%)     Naranja pálido #e8daef
```

### 7.2 Patrones de Componentes

#### **Card de Recomendación Rápida (Dashboard / Mobile)**

```
┌─────────────────────────────────────┐
│  GLD — Gold ETF   [ACTUALIZAR] [×]  │
├─────────────────────────────────────┤
│  🟢 BUY  |  Confianza 76%           │
│                                      │
│  Entrar en: 3–5 días               │
│  Salir en: 14 días                 │
│  P&L esperado: +$220 (+2.2%)       │
│  Máx. pérdida: -$230 (-2.3%)       │
│                                      │
│  [Ver detalles completos]  [Archiv] │
└─────────────────────────────────────┘
```

#### **Gauge de Confianza (Estilo Webreed/Stripe)**

```
         Baja      Media      Alta
          ↓         ↓         ↓
    ◄────────[█████░░░]───────►
              76% confianza

Explicación: "El modelo está seguro de
esta recomendación. Pero mercados son
inciertos. 24% de chance de sorpresa."
```

#### **Timeline Interactiva (Estilo Roadmap)**

```
Hoy   +2    +3    +4    +5          +14        +20
 │     │     │     │     │           │         │
 ●     ●     ●     ○     ○           ◆         ○
 │     │     │     │     │           │         │
BUY   BUY   BUY  HOLD  HOLD  ...   SELL    AVOID
       ↑                  ↑          ↑
    "Hoy"            "Óptimo"   "Cierra"
     76%              72%        65%
     conf             conf       conf
```

#### **Matrix de Scatter (Comparación de Activos)**

```
Eje Y: Max Return (P90)
Eje X: Max Risk (%)

Cada burbuja = 1 activo
Tamaño = Confianza
Color = BUY/HOLD/SELL/AVOID

       +8%
        │     BTC (6% riesgo, 72% conf)
        │      [Burbuja grande, verde]
      +6%
        │
      +4%
        │     GLD (2%, 76% conf)
        │      [Burbuja mediana, verde]
      +2%     SLV (3%, 60% conf)
        │      [Burbuja pequeña, naranja]
       0%
        │     PALL (-1%, 45% conf)
       -2%    [Burbuja pequeña, gris]
        └──────────────────────────
          0%   2%   4%   6%   8%
```

### 7.3 Interacciones & Microinteracciones

```
Acción                Marco                 Respuesta
──────────────────────────────────────────────────────
Hover en "Confianza"  Recomendación       Popover: "¿Confianza en qué?
                                           En que P50 es correcto.
                                           No es predicción de mercado."

Hover en "P10"        Escenarios card     Popover: "Worst case:
                                           1 de 10 veces. Sobre-pensar.
                                           Si ocurre, es x lo que entrenó."

Seleccionar modelo    Galería de modelos  Lado a lado con actual:
nuevo                                      compara RMSE, Sharpe, max DD

Presionar "Entrenar"  Botón                Transición a tab ENTRENAR
                                           Pre-cargado con último config

Arrastrar "Inversión" Slider               Actualiza P&L en tiempo real
$ 5k → 15k            en Recomendación    (Optimistic: +$230 → +$690)

Click en día del      Timeline             Expandible: precios, scores,
timeline                                   probabilidades, acciones

```

### 7.4 Estados de Carga y Vacío

```
CARGANDO:
┌─────────────────────────────────────┐
│ ⏳ Calculando escenarios...         │
│    Analizar fan chart (3–5 seg)    │
│                                      │
│ [████░░░░░░░░░░░░░░░░░░░░░░] 20%   │
└─────────────────────────────────────┘

SIN DATOS:
┌─────────────────────────────────────┐
│ 📊 Sin pronóstico aún               │
│                                      │
│ 1. Carga datos (tab DATA)           │
│ 2. Entrena o selecciona modelo      │
│ 3. Genera pronóstico                │
│                                      │
│ [Ir a DATA] [Tutorial]              │
└─────────────────────────────────────┘

ERROR:
┌─────────────────────────────────────┐
│ ❌ Error: Mercado cerrado          │
│    Intenta después de las 17:00     │
│                                      │
│ 📍 Última recomendación (12h atrás) │
│    [Cargar]                         │
└─────────────────────────────────────┘
```

---

## 8. Extensiones Futuras (No Bloqueantes)

### 8.1 Backtesting Simulado

```
Usuario selecciona:
  - Fecha histórica (ej. "1 de enero 2025")
  - Activo y modelo
  - Monto de inversión
  - TP%, SL%, horizonte

Sistema:
  - Ejecuta el modelo en esa fecha
  - Simula acima respetando el plan
  - Muestra P&L actual vs esperado
  - Diagnóstico: "¿Por qué falló?" o "¿Por qué ganó?"

Resultado:
  "Habrías entrado el 3 ene, salido el 15.
   Ganaste $324 (+3.2%).
   Pero el modelo predijo +2.2%.
   ¿Por qué la diferencia? Volatilidad menor."
```

### 8.2 Risk Budgeting

```
"He presupuestado $3,000 / mes en risk.
 Muestra qué posiciones encajan."

Sistema:
  - Recomienda posiciones por activo
  - Respeta max drawdown combinado
  - Optimiza Sharpe de cartera
  - Sugiere diversificación
```

### 8.3 Alertas por Email / Push

```
Usuario configura alertas:
  - "Notificar si BUY confirmation en GLD"
  - "Notificar si P10 toca stop-loss"
  - "Notificar cuando modelo necesite recalibración"

Sistema:
  - Monitorea condiciones diarias
  - Envía notificaciones con plantillas i18n
  - Incluye link a la app con contexto
```

### 8.4 Exportar Plan a CSV / PDF

```
Usuario genera recomendación
Presiona "Descargar Plan"

Formato CSV:
  Día, Acción, P10, P50, P90, Confianza, Rationale, ...

Formato PDF:
  Documento elegante con:
  - Resumen ejecutivo
  - Timeline ilustrada
  - Escenarios con gráficos
  - Rationales
  - Disclaimer de riesgo
  - Metadatos del modelo
```

### 8.5 Historial y Auditoría de Recomendaciones

```
"Archivo" → muestra todas las recomendaciones históricas
  - Mostrar qué se recomendó en cada fecha
  - Comparar resultado actual vs predicción
  - Calcular accuracy y Sharpe histórico
  - Identificar sesgos ("¿Siempre subestima volatilidad?")
```

---

## 9. No-Objetivos (Qué LA APP NO ES)

### Qué NO hacer, aunque sea tentador:

```
❌ Auto-trader
   App es un asistente de decisión.
   NO ejecuta trades automáticamente.
   Todo requiere confirmación humana.

❌ Black box de asesor financiero
   Tener que explicar rationales.
   Si no puedes entender la recomendación, no la ejecutes.

❌ App de criptos puro
   Soporta 4 activos, no es crypto casino.
   Mantener enfoque en análisis cuantitativo.

❌ Reemplazo para analista humano
   Es un herramienta de análisis.
   Para decisiones grandes, consulta profesionales.

❌ Predicción de mercado 100% confiable
   Modelos fallan. La volatilidad es incierta.
   Nunca garantizar resultados.

❌ Real-time quotes de trading
   30 min de retraso en datos es OK.
   No es una plataforma de micro-intraday.

❌ Herramienta de high-frequency trading
   Horizonte mínimo: 5 días.
   Targets: inversores con horizonte semanal–mensual.
```

---

## 10. Resumen de Cambios Organizacionales

### 10.1 Antes → Después

| Aspecto | Antes | Después |
|---------|-------|---------|
| **Entrada principal** | 8 pestañas iguales | Dashboard principal |
| **Pregunta de usuario** | "¿Cómo uso esto?" | "¿Debo invertir?" |
| **Flujo primario** | Data → Train → Forecast → Reco | Dashboard → Reco → Compare |
| **Entrenamiento** | Obligatorio, central | Opcional, avanzado |
| **Modelos** | Pestaña separada, confusa | Registry integrado, selección clara |
| **Pronóstico** | Fan chart desconectado | Fan chart vinculado a decisiones |
| **Recomendación** | Señal simple | Plan detallado + rationale + escenarios |
| **Tiempo a decisión** | 5–10 min | 30 seg (happy path) |
| **Usuarios objetivo** | Researchers | Inversores + Researchers |

### 10.2 Esfuerzo Estimado

```
Tarea                           Complejidad   Semanas
──────────────────────────────────────────────────
Rediseño de Dashboard          🟡 Media      2–3
Refactor Recomendación tab     🟡 Media      2–3
Crear tab Análisis (merge)     🟠 Moderada   1–2
Rediseño de Modelos            🟡 Media      1–2
Redefinir flujo Entrenar       🟡 Media      1–2
Mejorar Comparación            🟠 Moderada   1–2
Alertas/Notificaciones         🟠 Moderada   2–3 (Optional)
Backtesting                    🔴 Alta       3–4 (Optional)
Testing y QA                   🟡 Media      2–3
────────────────────────────────────────────────
TOTAL (Sin opcionales)         -             12–17 semanas
TOTAL (Con opcionales)         -             20–28 semanas
```

---

## 11. Conclusión

La aplicación tiene **excelente fundamento técnico** pero sufre de
**arquitectura UX confusa**. Los 8 tabs independientes sin jerarquía clara,
la obligación de entrenar modelos, y la desconexión entre pronóstico y decisión
crean **fricción innecesaria**.

### Cambios Propuestos (Resumen)

1. **Dashboard como entrada principal** — Responde "¿debo invertir?" en 30 seg
2. **Recomendación rediseñada** — Integra escenarios, rationale, timeline
3. **Entrenamiento como flujo opcional** — Para investigadores, no usuarios casuales
4. **Dos modos por defecto** — Inversor (simple) vs Investigador (técnico)
5. **Modelos como activos de larga vida** — Reutilizables, comparables, confiables
6. **Visualización de riesgo explícita** — P10/P50/P90 en dinero, no solo gráficos
7. **Flujos diferenciados** — Happy path (30 seg), Técnico (10 min), Portafolio (15 min)

### Valor Esperado

- **Nuevos usuarios:** Pueden tomar decisión en 30 segundos
- **Usuarios técnicos:** Tienen más profundidad, mejor organizada
- **Confianza:** Mayor claridad = mayor adopción
- **Diferenciación:** Se posiciona como "decisión-first", no "research-first"

---

## Apéndice: Mockups Conceptuales Simplificados

### A.1 Dashboard Propuesto (30 seg view)

```
═══════════════════════════════════════════════════════════════════
                    MULTI-ASSET DECISION BOARD
═══════════════════════════════════════════════════════════════════

🌍 Seleccionar Activo: [GLD ▼]   📈 Período: [20 días ▼]   💰 Inversión: [$ 10,000]

───────────────────────────────────────────────────────────────────

                        🟢 GLD — BUY
                     Confianza: 76%

     Entrada: 3–5 días  |  Salida: 14 días  |  P&L: +$220 (+2.2%)

     [Ver Detalles] [Por Qué] [Escenarios] [COMPARAR]

───────────────────────────────────────────────────────────────────

                    LEADERBOARD (Todos los activos)

     Ranking │ Activo  │ Acción │ Conf.│ P&L Esp. │ Máx. Riesgo
     ────────┼─────────┼────────┼──────┼──────────┼────────────
       1     │ BTC     │ 🟢 BUY │ 72% │ +8.2%    │  -3.1%
       2     │ GLD     │ 🟢 BUY │ 76% │ +2.2%    │  -2.3%
       3     │ SLV     │ 🟡 HOL │ 60% │ +1.5%    │  -2.4%
       4     │ PALL    │ 🔴 AVD │ 45% │ -0.5%    │  -3.2%

═══════════════════════════════════════════════════════════════════
                    ℹ️ ULTIMO ACTUALIZADO: hoy 16:45
═══════════════════════════════════════════════════════════════════
```

### A.2 Rec Detail Propuesto (3 min view)

```
═══════════════════════════════════════════════════════════════════
                    GLD — RECOMENDACIÓN DETALLADA
═══════════════════════════════════════════════════════════════════

🟢 BUY HOYMENTE  |  Confianza 76%

Narrative (plain English):
   "SMA-50 cruzó SMA-200 al alza. ATR en range normal.
    P10 drawdown (-2.3%) está dentro de tu SL (-3%). 
    Hoy es óptimo para entrar; acumula en los próx. 3–5 días."

───────────────────────────────────────────────────────────────────

PLAN DE ACCIÓN (Timeline interactiva):

    Hoy    +2d    +3d    +4d    +5d   ...  +14d   +20d
     │      │      │      │      │         │      │
     ●      ●      ●      ○      ○         ◆      ○
     │      │      │      │      │         │      │
    BUY    BUY    BUY   HOLD   HOLD  ...  SELL   AVOID
    76%    75%    74%    73%    72%        70%    60%
    conf   conf   conf   conf   conf       conf   conf

    ☑️ Hoy es dentro de la "ventana de entrada"
    ☑️ Máxima ganancia esperada: día +14 (median: +2.8%)

───────────────────────────────────────────────────────────────────

RATIONALE DE 4 FACTORES:

[▼] 1️⃣  Tendencia (SMA-50/200)
    🟢 SMA-50 > SMA-200  [Golden Cross activo desde 8 días]
    Señal: ALCISTA

[▼] 2️⃣  Volatilidad (ATR%)
    🟡 ATR = 1.8%  [Rango normal para GLD: 1.2–2.4%]
    Señal: NEUTRAL — volatividad predecible

[▼] 3️⃣  Riesgo de Cuantil (P10 drawdown)
    🟢 P10: -2.3% vs Tu SL: -3.0%  [Margen: +0.7%]
    Señal: SEGURO — pérdida máxima dentro de límite

[▼] 4️⃣  Evaluación de Hoy
    🟢 Score técnico: 0.76 / 1.0
    Señal: HOY ES ÓPTIMO  [Dentro de ventana de entrada]

───────────────────────────────────────────────────────────────────

TRES ESCENARIOS (Inversión $10,000):

┌────────────────────────────────────────────────────┐
│ 🔴 PESIMISTA (P10 — 10% de probabilidad)          │
│                                                    │
│    Precio inicial: $195.00                        │
│    Precio final (día 20): $190.54                 │
│    Cambio: -2.3%  →  P&L: -$230                  │
│    Máxima ganancia intermedia: +0.5% (día 8)     │
│                                                    │
│    Interpretación: "Mejor esperar, mercado niega" │
├────────────────────────────────────────────────────┤
│ 🟡 BASE (P50 — mediana, si decides hoy)          │
│                                                    │
│    Precio inicial: $195.00                        │
│    Precio final (día 20): $199.21                 │
│    Cambio: +2.2%  →  P&L: +$220                  │
│    Máxima ganancia: +2.8% (día 14) ← SALIR AQUÍ  │
│                                                    │
│    Interpretación: "Plan de acción funciona       │
│                     como se espera"               │
├────────────────────────────────────────────────────┤
│ 🟢 OPTIMISTA (P90 — 10% de probabilidad)          │
│                                                    │
│    Precio inicial: $195.00                        │
│    Precio final (día 20): $207.35                 │
│    Cambio: +6.4%  →  P&L: +$640                  │
│    Máxima ganancia: +7.2% (día 18)               │
│                                                    │
│    Interpretación: "Mejor de lo esperado. Rare."  │
└────────────────────────────────────────────────────┘

RESUMEN DE RIESGO:
  Resultado esperado (P50): +$220 (+2.2%)
  Rango probable (P10–P90): -$230 to +$640
  Ratio riesgo/recompensa: 1:2.8 ✓ BUENO
  Máxima pérdida esperada: -2.3% (dentro de límite)
  Confianza en P50: 76% (buena)

───────────────────────────────────────────────────────────────────

WARNINGS & CONTEXTO:

⚠️  Volatilidad en máximos de 6 meses → Aumenta incertidumbre
💡  Modelo fue entrenado hace 8 días → Recalibración en 3 días
📊  Accuracy histórica: 52% ganador, 48% perdedor (modesto)
ℹ️  Datos ~30 min atrasados (último update: 16:45 hoy)

───────────────────────────────────────────────────────────────────

[📊 Ver Fan Chart Detallado] [⚙️ Ajustar Parámetros]
[💾 Archivar Plan]          [➕ Agregar a Cartera]

═══════════════════════════════════════════════════════════════════
```

---

**FIN DEL DOCUMENTO**

**Próximos Pasos:**
1. Revisión y feedback de stakeholders
2. Priorización de features (MVP vs Nice-to-Have)
3. Diseño detallado de componentes (Figma / wireframes)
4. Desarrollo iterativo (2–4 semanas por sprint)
5. Validación con usuarios (A/B testing opcional)
