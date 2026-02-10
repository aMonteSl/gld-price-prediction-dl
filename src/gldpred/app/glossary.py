"""Educational glossary with reusable popover component.

Provides ``info_term(label, term_key)`` — a Streamlit popover that shows
a short educational explanation next to any metric or parameter in the UI.
All glossary content is bilingual (EN / ES) and keyed by term slug.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import streamlit as st


@dataclass(frozen=True)
class GlossaryEntry:
    """A single glossary term in one language."""

    title: str
    explanation: str
    why_it_matters: str
    example: str = ""


# ── Glossary data (EN / ES) ─────────────────────────────────────────────
GLOSSARY: Dict[str, Dict[str, GlossaryEntry]] = {
    # ----- Model & training parameters --------------------------------
    "architecture": {
        "en": GlossaryEntry(
            title="Architecture",
            explanation="The neural-network design used for forecasting. TCN uses causal convolutions, GRU/LSTM are recurrent.",
            why_it_matters="Different architectures trade off speed vs. ability to capture long-range patterns.",
            example="TCN is fastest; LSTM retains more context for very long sequences.",
        ),
        "es": GlossaryEntry(
            title="Arquitectura",
            explanation="El diseño de red neuronal usado para el pronóstico. TCN usa convoluciones causales, GRU/LSTM son recurrentes.",
            why_it_matters="Diferentes arquitecturas equilibran velocidad vs. capacidad para captar patrones de largo alcance.",
            example="TCN es la más rápida; LSTM retiene más contexto en secuencias muy largas.",
        ),
    },
    "hidden_size": {
        "en": GlossaryEntry(
            title="Hidden Size",
            explanation="Number of neurons in each hidden layer of the network.",
            why_it_matters="Larger = more capacity to learn complex patterns but slower and more prone to overfitting.",
            example="64 is a good default; try 128 for complex assets like BTC-USD.",
        ),
        "es": GlossaryEntry(
            title="Tamaño Oculto",
            explanation="Número de neuronas en cada capa oculta de la red.",
            why_it_matters="Mayor = más capacidad para aprender patrones complejos, pero más lento y propenso al sobreajuste.",
            example="64 es un buen predeterminado; pruebe 128 para activos complejos como BTC-USD.",
        ),
    },
    "num_layers": {
        "en": GlossaryEntry(
            title="Number of Layers",
            explanation="How many hidden layers are stacked in the network.",
            why_it_matters="More layers = deeper network capable of abstraction, but harder to train.",
            example="2 layers work well for most assets.",
        ),
        "es": GlossaryEntry(
            title="Número de Capas",
            explanation="Cuántas capas ocultas se apilan en la red.",
            why_it_matters="Más capas = red más profunda capaz de abstracción, pero más difícil de entrenar.",
            example="2 capas funcionan bien para la mayoría de activos.",
        ),
    },
    "epochs": {
        "en": GlossaryEntry(
            title="Epochs",
            explanation="Number of complete passes through the training data.",
            why_it_matters="Too few → underfitting; too many → overfitting. Watch the loss curve.",
            example="Start with 50; reduce if validation loss rises.",
        ),
        "es": GlossaryEntry(
            title="Épocas",
            explanation="Número de pasadas completas por los datos de entrenamiento.",
            why_it_matters="Muy pocas → infraajuste; demasiadas → sobreajuste. Observe la curva de pérdida.",
            example="Empiece con 50; reduzca si la pérdida de validación sube.",
        ),
    },
    "batch_size": {
        "en": GlossaryEntry(
            title="Batch Size",
            explanation="Number of samples processed before updating model weights.",
            why_it_matters="Larger batches give smoother gradients but use more memory.",
            example="32 is a balanced default.",
        ),
        "es": GlossaryEntry(
            title="Tamaño de Lote",
            explanation="Número de muestras procesadas antes de actualizar los pesos del modelo.",
            why_it_matters="Lotes mayores dan gradientes más suaves pero usan más memoria.",
            example="32 es un predeterminado equilibrado.",
        ),
    },
    "learning_rate": {
        "en": GlossaryEntry(
            title="Learning Rate",
            explanation="Step size for weight updates during training.",
            why_it_matters="Too high → unstable training; too low → very slow convergence.",
            example="0.001 is the standard starting point.",
        ),
        "es": GlossaryEntry(
            title="Tasa de Aprendizaje",
            explanation="Tamaño del paso para actualizar los pesos durante el entrenamiento.",
            why_it_matters="Muy alta → entrenamiento inestable; muy baja → convergencia muy lenta.",
            example="0.001 es el punto de partida estándar.",
        ),
    },
    "seq_length": {
        "en": GlossaryEntry(
            title="Sequence Length",
            explanation="Number of past trading days the model looks at to make a prediction.",
            why_it_matters="Longer sequences capture more history but increase compute cost.",
            example="20 days ≈ 1 trading month.",
        ),
        "es": GlossaryEntry(
            title="Longitud de Secuencia",
            explanation="Número de días de negociación pasados que el modelo observa para hacer una predicción.",
            why_it_matters="Secuencias más largas capturan más historia pero aumentan el costo de cómputo.",
            example="20 días ≈ 1 mes de negociación.",
        ),
    },
    "forecast_steps": {
        "en": GlossaryEntry(
            title="Forecast Steps (K)",
            explanation="How many future trading days the model predicts.",
            why_it_matters="More steps = further-out forecast, but uncertainty grows with distance.",
            example="K=20 gives roughly a one-month outlook.",
        ),
        "es": GlossaryEntry(
            title="Pasos de Pronóstico (K)",
            explanation="Cuántos días de negociación futuros predice el modelo.",
            why_it_matters="Más pasos = pronóstico más lejano, pero la incertidumbre crece con la distancia.",
            example="K=20 da una perspectiva de aproximadamente un mes.",
        ),
    },
    # ----- Metrics & evaluation ----------------------------------------
    "pinball_loss": {
        "en": GlossaryEntry(
            title="Pinball (Quantile) Loss",
            explanation="Asymmetric loss function that penalises under- and over-prediction differently for each quantile.",
            why_it_matters="Produces well-calibrated uncertainty bands (P10/P50/P90).",
        ),
        "es": GlossaryEntry(
            title="Pérdida Pinball (Cuantílica)",
            explanation="Función de pérdida asimétrica que penaliza la sub- y sobre-predicción de manera diferente para cada cuantil.",
            why_it_matters="Produce bandas de incertidumbre bien calibradas (P10/P50/P90).",
        ),
    },
    "directional_accuracy": {
        "en": GlossaryEntry(
            title="Directional Accuracy",
            explanation="Fraction of days where the model correctly predicts whether the return is positive or negative.",
            why_it_matters="Even a small edge over 50 % can be valuable for trading decisions.",
        ),
        "es": GlossaryEntry(
            title="Precisión Direccional",
            explanation="Fracción de días donde el modelo predice correctamente si el rendimiento es positivo o negativo.",
            why_it_matters="Incluso una pequeña ventaja sobre 50 % puede ser valiosa para decisiones de inversión.",
        ),
    },
    "mse": {
        "en": GlossaryEntry(
            title="MSE (Mean Squared Error)",
            explanation="Average of squared prediction errors. Heavily penalises large mistakes.",
            why_it_matters="Lower is better. Compare across models trained on the same asset.",
        ),
        "es": GlossaryEntry(
            title="MSE (Error Cuadrático Medio)",
            explanation="Promedio de los errores de predicción al cuadrado. Penaliza fuertemente los errores grandes.",
            why_it_matters="Menor es mejor. Compare entre modelos entrenados sobre el mismo activo.",
        ),
    },
    "calibration": {
        "en": GlossaryEntry(
            title="Quantile Calibration",
            explanation="Checks whether the P10 band really contains ~10 % of outcomes, P90 ~90 %, etc.",
            why_it_matters="Well-calibrated bands make the uncertainty estimate trustworthy.",
        ),
        "es": GlossaryEntry(
            title="Calibración Cuantílica",
            explanation="Verifica si la banda P10 realmente contiene ~10 % de los resultados, P90 ~90 %, etc.",
            why_it_matters="Bandas bien calibradas hacen que la estimación de incertidumbre sea confiable.",
        ),
    },
    # ----- Decision / Risk terms ---------------------------------------
    "stop_loss": {
        "en": GlossaryEntry(
            title="Stop-Loss",
            explanation="The maximum percentage drop at which you should exit a position to limit losses.",
            why_it_matters="Protects against catastrophic drawdowns.",
            example="A stop-loss of −3.5 % means sell if price drops 3.5 % from entry.",
        ),
        "es": GlossaryEntry(
            title="Stop-Loss",
            explanation="El porcentaje máximo de caída al que debería salir de una posición para limitar pérdidas.",
            why_it_matters="Protege contra caídas catastróficas.",
            example="Un stop-loss de −3.5 % significa vender si el precio cae 3.5 % desde la entrada.",
        ),
    },
    "take_profit": {
        "en": GlossaryEntry(
            title="Take-Profit",
            explanation="Target percentage gain at which to close a position and lock in profits.",
            why_it_matters="Avoids giving back gains by setting a disciplined exit point.",
        ),
        "es": GlossaryEntry(
            title="Take-Profit",
            explanation="Porcentaje objetivo de ganancia al cual cerrar una posición y asegurar beneficios.",
            why_it_matters="Evita devolver ganancias al establecer un punto de salida disciplinado.",
        ),
    },
    "risk_reward": {
        "en": GlossaryEntry(
            title="Risk / Reward Ratio",
            explanation="Potential upside divided by potential downside. Values > 1 mean more reward than risk.",
            why_it_matters="A ratio of 2 : 1 means you risk $1 to potentially gain $2.",
        ),
        "es": GlossaryEntry(
            title="Ratio Riesgo / Beneficio",
            explanation="Potencial alcista dividido por potencial bajista. Valores > 1 significan más beneficio que riesgo.",
            why_it_matters="Un ratio de 2 : 1 significa que arriesga $1 para potencialmente ganar $2.",
        ),
    },
    "max_drawdown": {
        "en": GlossaryEntry(
            title="Max Drawdown",
            explanation="Largest peak-to-trough decline in the forecast trajectory.",
            why_it_matters="Indicates worst-case intra-period loss you might experience.",
        ),
        "es": GlossaryEntry(
            title="Drawdown Máximo",
            explanation="Mayor declive de pico a valle en la trayectoria pronosticada.",
            why_it_matters="Indica la peor pérdida intraperiodo que podría experimentar.",
        ),
    },
    "market_regime": {
        "en": GlossaryEntry(
            title="Market Regime",
            explanation="Characterisation of the current market state — trending up/down, ranging, or high-volatility.",
            why_it_matters="Strategy effectiveness varies by regime; trending markets favour momentum, ranging markets favour mean-reversion.",
        ),
        "es": GlossaryEntry(
            title="Régimen de Mercado",
            explanation="Caracterización del estado actual del mercado — tendencia alcista/bajista, lateral o alta volatilidad.",
            why_it_matters="La efectividad de la estrategia varía por régimen; mercados en tendencia favorecen momento, laterales favorecen reversión a la media.",
        ),
    },
    "confidence": {
        "en": GlossaryEntry(
            title="Confidence Score",
            explanation="A 0–100 score summarising how strongly the model signals BUY, HOLD, or AVOID.",
            why_it_matters="Higher confidence = more agreement among the five scoring signals.",
        ),
        "es": GlossaryEntry(
            title="Puntuación de Confianza",
            explanation="Una puntuación 0–100 que resume cuán fuertemente el modelo señala COMPRAR, MANTENER o EVITAR.",
            why_it_matters="Mayor confianza = más acuerdo entre las cinco señales de puntuación.",
        ),
    },
    # ----- Diagnostics -------------------------------------------------
    "overfitting": {
        "en": GlossaryEntry(
            title="Overfitting",
            explanation="The model memorises training data instead of learning generalisable patterns. Validation loss rises while training loss falls.",
            why_it_matters="An overfit model performs well on old data but poorly on new data.",
        ),
        "es": GlossaryEntry(
            title="Sobreajuste",
            explanation="El modelo memoriza los datos de entrenamiento en lugar de aprender patrones generalizables. La pérdida de validación sube mientras la de entrenamiento baja.",
            why_it_matters="Un modelo sobreajustado funciona bien en datos antiguos pero mal en datos nuevos.",
        ),
    },
    "underfitting": {
        "en": GlossaryEntry(
            title="Underfitting",
            explanation="The model is too simple or undertrained to capture the data's patterns. Both losses remain high.",
            why_it_matters="An underfit model makes poor predictions on all data.",
        ),
        "es": GlossaryEntry(
            title="Infraajuste",
            explanation="El modelo es demasiado simple o poco entrenado para captar los patrones de los datos. Ambas pérdidas permanecen altas.",
            why_it_matters="Un modelo infraajustado hace predicciones pobres en todos los datos.",
        ),
    },
    # ----- Chart / Forecast terms --------------------------------------
    "fan_chart": {
        "en": GlossaryEntry(
            title="Fan Chart",
            explanation="A visualisation showing the median forecast (P50) with shaded uncertainty bands (P10–P90).",
            why_it_matters="Wider bands = more uncertainty; helps you gauge how reliable the forecast is.",
        ),
        "es": GlossaryEntry(
            title="Gráfico de Abanico",
            explanation="Visualización que muestra el pronóstico mediano (P50) con bandas de incertidumbre sombreadas (P10–P90).",
            why_it_matters="Bandas más anchas = más incertidumbre; ayuda a evaluar cuán confiable es el pronóstico.",
        ),
    },
    "quantiles": {
        "en": GlossaryEntry(
            title="Quantiles (P10 / P50 / P90)",
            explanation="P10 = 10th-percentile (pessimistic), P50 = median (central), P90 = 90th-percentile (optimistic).",
            why_it_matters="They express the range of likely outcomes rather than a single point forecast.",
        ),
        "es": GlossaryEntry(
            title="Cuantiles (P10 / P50 / P90)",
            explanation="P10 = percentil 10 (pesimista), P50 = mediana (central), P90 = percentil 90 (optimista).",
            why_it_matters="Expresan el rango de resultados probables en lugar de un pronóstico puntual.",
        ),
    },
    "sma": {
        "en": GlossaryEntry(
            title="SMA (Simple Moving Average)",
            explanation="The arithmetic mean of the closing price over the last N days.",
            why_it_matters="SMA-50 crossing above SMA-200 (golden cross) is a classic bullish signal.",
            example="SMA-200 smooths out noise to reveal the long-term trend.",
        ),
        "es": GlossaryEntry(
            title="SMA (Media Móvil Simple)",
            explanation="La media aritmética del precio de cierre en los últimos N días.",
            why_it_matters="SMA-50 cruzando por encima de SMA-200 (cruce dorado) es una señal alcista clásica.",
            example="SMA-200 suaviza el ruido para revelar la tendencia a largo plazo.",
        ),
    },
    "atr": {
        "en": GlossaryEntry(
            title="ATR (Average True Range)",
            explanation="Measures price volatility as the average of daily high–low ranges (adjusted for gaps).",
            why_it_matters="High ATR% means higher volatility, which increases both opportunity and risk.",
        ),
        "es": GlossaryEntry(
            title="ATR (Rango Verdadero Medio)",
            explanation="Mide la volatilidad del precio como el promedio de rangos diarios máximo–mínimo (ajustado por brechas).",
            why_it_matters="ATR% alto significa mayor volatilidad, lo que aumenta tanto la oportunidad como el riesgo.",
        ),
    },
    "rsi": {
        "en": GlossaryEntry(
            title="RSI (Relative Strength Index)",
            explanation="Momentum oscillator (0–100). Above 70 = overbought, below 30 = oversold.",
            why_it_matters="Helps identify potential reversal points.",
        ),
        "es": GlossaryEntry(
            title="RSI (Índice de Fuerza Relativa)",
            explanation="Oscilador de momento (0–100). Por encima de 70 = sobrecomprado, por debajo de 30 = sobrevendido.",
            why_it_matters="Ayuda a identificar puntos de posible reversión.",
        ),
    },
}


def info_term(label: str, term_key: str, lang: str = "en") -> None:
    """Render a metric label with an ℹ️ educational popover.

    If ``term_key`` is not in the glossary the label is rendered plain.

    Args:
        label: Display text shown inline (e.g. "Hidden Size").
        term_key: Slug matching a key in ``GLOSSARY``.
        lang: Language code (``"en"`` or ``"es"``).
    """
    entry_map = GLOSSARY.get(term_key)
    if not entry_map:
        st.markdown(f"**{label}**")
        return

    entry = entry_map.get(lang, entry_map.get("en"))
    if entry is None:
        st.markdown(f"**{label}**")
        return

    with st.popover(f"ℹ️ {label}"):
        st.markdown(f"### {entry.title}")
        st.markdown(entry.explanation)
        st.markdown(f"**↳ {_why_label(lang)}** {entry.why_it_matters}")
        if entry.example:
            st.markdown(f"*{entry.example}*")


def _why_label(lang: str) -> str:
    return "¿Por qué importa?" if lang == "es" else "Why it matters:"
