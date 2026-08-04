"""
Clasificador de Literatura Médica
---------------------------------
App de demostración (Gradio) para un modelo de clasificación multiclase
(TF-IDF + Regresión Logística) que predice la categoría médica de un
artículo a partir de su título y resumen (abstract).

Pensada para desplegarse gratis en Hugging Face Spaces.
"""

import json
import os
import re

import gradio as gr
import joblib
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# 1. Carga de artefactos (se ejecuta una sola vez al iniciar la app)
# ---------------------------------------------------------------------
vectorizer = joblib.load("vectorizer.joblib")
model = joblib.load("medical_pipeline.joblib")

with open("metrics.json", "r", encoding="utf-8") as f:
    METRICS = json.load(f)

# Orden alfabético de las 15 categorías (coincide con el orden usado
# al entrenar el modelo: model.classes_ = [0..14] -> este listado)
LABELS = sorted(METRICS["per_class_f1"].keys())

# Stopwords en inglés incrustadas directamente (evita depender de una
# descarga de NLTK en tiempo de ejecución, lo cual puede fallar o
# ralentizar el arranque en un servidor gratuito).
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "ain", "all", "am", "an",
    "and", "any", "are", "aren", "aren't", "as", "at", "be", "because", "been",
    "before", "being", "below", "between", "both", "but", "by", "can", "couldn",
    "couldn't", "d", "did", "didn", "didn't", "do", "does", "doesn", "doesn't",
    "doing", "don", "don't", "down", "during", "each", "few", "for", "from",
    "further", "had", "hadn", "hadn't", "has", "hasn", "hasn't", "have", "haven",
    "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "hers",
    "herself", "him", "himself", "his", "how", "i", "i'd", "i'll", "i'm", "i've",
    "if", "in", "into", "is", "isn", "isn't", "it", "it'd", "it'll", "it's",
    "its", "itself", "just", "ll", "m", "ma", "me", "mightn", "mightn't", "more",
    "most", "mustn", "mustn't", "my", "myself", "needn", "needn't", "no", "nor",
    "not", "now", "o", "of", "off", "on", "once", "only", "or", "other", "our",
    "ours", "ourselves", "out", "over", "own", "re", "s", "same", "shan",
    "shan't", "she", "she'd", "she'll", "she's", "should", "should've",
    "shouldn", "shouldn't", "so", "some", "such", "t", "than", "that", "that'll",
    "the", "their", "theirs", "them", "themselves", "then", "there", "these",
    "they", "they'd", "they'll", "they're", "they've", "this", "those",
    "through", "to", "too", "under", "until", "up", "ve", "very", "was", "wasn",
    "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren",
    "weren't", "what", "when", "where", "which", "while", "who", "whom", "why",
    "will", "with", "won", "won't", "wouldn", "wouldn't", "y", "you", "you'd",
    "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves",
}


def limpiar_texto(texto: str) -> str:
    """Misma limpieza usada en el entrenamiento: minúsculas, solo letras,
    espacios normalizados y stopwords en inglés removidas."""
    if not isinstance(texto, str):
        return ""
    texto = texto.lower()
    texto = re.sub(r"[^a-z\s]", "", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    palabras = texto.split()
    return " ".join(w for w in palabras if w not in STOPWORDS)


# ---------------------------------------------------------------------
# 2. Función de inferencia
# ---------------------------------------------------------------------
def predecir(title: str, abstract: str):
    if not title and not abstract:
        return "Escribe al menos un título o un resumen.", {}

    texto_completo = f"{title or ''} {abstract or ''}"
    texto_limpio = limpiar_texto(texto_completo)

    if not texto_limpio:
        return "El texto no contiene palabras reconocibles tras la limpieza.", {}

    X = vectorizer.transform([texto_limpio])

    pred_idx = model.predict(X)[0]
    categoria = LABELS[pred_idx]

    # Probabilidades por clase (LogisticRegression con solver liblinear
    # soporta predict_proba)
    probs = model.predict_proba(X)[0]
    prob_dict = {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}
    # Gradio Label espera un dict ordenado de mayor a menor
    prob_dict = dict(sorted(prob_dict.items(), key=lambda x: x[1], reverse=True))

    return categoria, prob_dict


# ---------------------------------------------------------------------
# 3. Interfaz Gradio
# ---------------------------------------------------------------------
EJEMPLOS = [
    [
        "Left ventricular remodeling after acute myocardial infarction",
        "This study evaluates changes in left ventricular structure and "
        "function following acute myocardial infarction, using serial "
        "echocardiographic assessment in a cohort of patients treated with "
        "early reperfusion therapy.",
    ],
    [
        "Cognitive decline and biomarkers in early-stage Alzheimer's disease",
        "We investigated the association between cerebrospinal fluid "
        "biomarkers and longitudinal cognitive decline in patients with "
        "early-stage Alzheimer's disease.",
    ],
]

with gr.Blocks(title="Clasificador de Literatura Médica") as demo:
    gr.Markdown(
        """
        # 🩺 Clasificador de Literatura Médica
        Modelo de **TF-IDF + Regresión Logística** que clasifica artículos
        médicos en una o varias de estas categorías: **cardiovascular,
        hepatorenal, neurológico y oncológico** (y combinaciones entre ellas),
        a partir del título y el resumen (abstract) del artículo.
        """
    )

    with gr.Tab("Predecir"):
        with gr.Row():
            with gr.Column():
                title_in = gr.Textbox(label="Título del artículo", lines=2)
                abstract_in = gr.Textbox(label="Abstract / Resumen", lines=8)
                btn = gr.Button("Clasificar", variant="primary")
                gr.Examples(examples=EJEMPLOS, inputs=[title_in, abstract_in])
            with gr.Column():
                categoria_out = gr.Textbox(label="Categoría predicha")
                probs_out = gr.Label(label="Probabilidad por categoría", num_top_classes=6)

        btn.click(predecir, inputs=[title_in, abstract_in], outputs=[categoria_out, probs_out])
        title_in.submit(predecir, inputs=[title_in, abstract_in], outputs=[categoria_out, probs_out])
        abstract_in.submit(predecir, inputs=[title_in, abstract_in], outputs=[categoria_out, probs_out])

    with gr.Tab("Sobre el proyecto"):
        gr.Markdown(
            f"""
            ### Métricas del modelo (conjunto de prueba)
            - **Accuracy:** {METRICS['accuracy']:.2%}
            - **F1 macro:** {METRICS['macro_f1']:.2%}
            - **F1 ponderado:** {METRICS['weighted_f1']:.2%}

            El dataset está desbalanceado (ver gráfico de distribución de
            clases abajo), por lo que el F1 macro es más bajo que el
            accuracy: el modelo predice mucho mejor las categorías con más
            ejemplos (neurological, cardiovascular, hepatorenal,
            oncological) que las combinaciones minoritarias.
            """
        )
        with gr.Row():
            gr.Image("class_distribution.png", label="Distribución de clases", show_label=True)
            gr.Image("text_length_distribution.png", label="Distribución de longitud de texto", show_label=True)

if __name__ == "__main__":
    # Render (y la mayoría de PaaS gratuitos) asignan el puerto a través
    # de la variable de entorno PORT. En local, si no existe, usa 7860.
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
