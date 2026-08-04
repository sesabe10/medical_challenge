# Clasificador de Literatura Médica

Demo de un modelo de clasificación de texto (TF-IDF + Regresión Logística)
que predice la categoría médica de un artículo científico (cardiovascular,
hepatorenal, neurológico, oncológico, o combinaciones de estas) a partir
de su título y resumen.

App construida con [Gradio](https://gradio.app).

## Cómo desplegar esto gratis en Render

Render tiene un plan gratuito real para aplicaciones web en Python
(incluye Gradio). El único costo es que el servicio "duerme" tras ~15
minutos sin visitas y tarda 30-60 segundos en despertar la primera vez
que alguien entra — totalmente normal para un link de portafolio.

### Paso 1: Sube este proyecto a GitHub

Si no tienes el proyecto en GitHub todavía:

1. Crea una cuenta gratis en https://github.com si no tienes una.
2. Crea un repositorio nuevo (botón verde "New") — puede ser público o privado, ej. `clasificador-literatura-medica`. No marques "Add a README" (ya tienes uno).
3. En tu computador, dentro de esta carpeta (`deploy/`), corre:

```bash
git init
git add .
git commit -m "Deploy inicial"
git branch -M main
git remote add origin https://github.com/TU_USUARIO/clasificador-literatura-medica.git
git push -u origin main
```

(Reemplaza `TU_USUARIO` por tu usuario de GitHub. Si te pide autenticación, GitHub ya no acepta contraseña normal — usa un [Personal Access Token](https://github.com/settings/tokens) como contraseña, o `gh auth login` si tienes la CLI de GitHub instalada.)

### Paso 2: Crea el servicio en Render

1. Crea una cuenta gratis en https://render.com (puedes entrar directo con tu cuenta de GitHub).
2. Click en **New +** → **Web Service**.
3. Conecta tu repositorio de GitHub (`clasificador-literatura-medica`). Si no aparece, dale permiso a Render sobre ese repo desde la pantalla de conexión.
4. Render debería detectar automáticamente el archivo `render.yaml` y preconfigurar todo. Si no, configura manualmente:
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python app.py`
   - **Instance Type**: `Free`
5. Click en **Create Web Service**.
6. Espera unos minutos mientras Render instala dependencias y arranca la app (verás los logs en vivo).
7. Cuando termine, tu app estará disponible en una URL tipo:
   `https://clasificador-literatura-medica.onrender.com`

Ese es el link que puedes poner en tu portafolio o CV.

### Actualizaciones futuras

Cada vez que hagas `git push` a la rama `main`, Render vuelve a desplegar
automáticamente la app con los cambios.

## Correrlo localmente antes de desplegar (opcional pero recomendado)

```bash
pip install -r requirements.txt
python app.py
```

Esto abre la app en `http://127.0.0.1:7860`.

## Archivos de este proyecto

- `app.py` — la aplicación (interfaz Gradio + lógica de inferencia).
- `requirements.txt` — dependencias de Python, ya probadas.
- `render.yaml` — configuración para que Render despliegue automáticamente.
- `runtime.txt` — versión de Python a usar en Render.
- `vectorizer.joblib`, `medical_pipeline.joblib` — artefactos del modelo entrenado.
- `metrics.json` — métricas del modelo (accuracy, F1 por clase), usadas en la pestaña "Sobre el proyecto".
- `class_distribution.png`, `text_length_distribution.png` — gráficos exploratorios del dataset.

## Nota sobre el orden de las categorías

El modelo fue entrenado con las categorías codificadas como enteros
(0-14). Este `app.py` reconstruye el orden correcto a partir de las
claves de `metrics.json` (orden alfabético), que es el mismo orden usado
por `LabelEncoder` durante el entrenamiento. Si vuelves a entrenar el
modelo, asegúrate de mantener ese mismo orden o actualiza `LABELS` en
`app.py`.
