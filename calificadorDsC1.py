import streamlit as st
import sqlite3
import pdfplumber
import re
import requests
import pandas as pd
from io import BytesIO
import os
import json
from datetime import datetime

# Si usas Gemini dejar la configuración, sino comenta estas líneas
from google import genai

# ----------------------------
# CONFIGURACIÓN GEMINI
# ----------------------------
client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
gemini_model = "gemini-2.0-flash"

# ----------------------------
# RUTA DB y CREACIÓN TABLAS
# ----------------------------
DB_PATH = "badges.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # tabla badges (existente)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS badges (
        url TEXT PRIMARY KEY,
        id TEXT,
        nrc TEXT,
        nombre TEXT,
        curso TEXT,
        horas TEXT,
        fecha TEXT,
        valido INTEGER,
        mensaje TEXT,
        nota_taller REAL
    )
    """)
    # tabla para resultados de taller (nueva)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS taller_resultados (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        id_estudiante TEXT,
        nrc_curso TEXT,
        nota_taller REAL,
        feedback TEXT,
        comentarios TEXT,
        fecha_registro TEXT
    )
    """)
    # tabla respuestas_cuestionario (aseguramos que exista)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS respuestas_cuestionario (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        id_estudiante TEXT,
        nrc_curso TEXT,
        respuestas TEXT,
        nota_cuestionario REAL,
        feedback TEXT,
        fecha_registro TEXT
    )
    """)
    conn.commit()
    conn.close()

# ----------------------------
# FUNCIONES DE BASE DE DATOS
# ----------------------------
def guardar_en_db(id_est, nrc, nombre, curso, horas, fecha, url, valido, mensaje, nota_taller=None):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
    INSERT OR REPLACE INTO badges (url, id, nrc, nombre, curso, horas, fecha, valido, mensaje, nota_taller)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (url, id_est, nrc, nombre, curso, horas, fecha, int(bool(valido)), mensaje, nota_taller))
    conn.commit()
    conn.close()

def obtener_registros():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM badges", conn)
    conn.close()
    return df

def obtener_calificaciones_badges():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT 
            id AS id_estudiante,
            nrc AS nrc_curso,
            SUM(CASE WHEN valido=1 THEN 1 ELSE 0 END) * 0.25 AS calificacion_badge
        FROM badges
        GROUP BY id, nrc
    """, conn)
    conn.close()
    return df

def guardar_resultado_taller(id_estudiante, nrc_curso, nota_taller, feedback, comentarios):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO taller_resultados (id_estudiante, nrc_curso, nota_taller, feedback, comentarios, fecha_registro)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (id_estudiante, nrc_curso, nota_taller, feedback, comentarios, datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()

def obtener_aggregado_taller():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT 
            id_estudiante,
            nrc_curso,
            SUM(COALESCE(nota_taller,0)) AS calificacion_taller,
            COUNT(*) AS n_registros_taller
        FROM taller_resultados
        GROUP BY id_estudiante, nrc_curso
    """, conn)
    conn.close()
    return df

def guardar_respuestas_cuestionario(id_estudiante, nrc_curso, respuestas_dict, nota_cuestionario, feedback):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO respuestas_cuestionario (id_estudiante, nrc_curso, respuestas, nota_cuestionario, feedback, fecha_registro)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        id_estudiante,
        nrc_curso,
        json.dumps(respuestas_dict, ensure_ascii=False),
        nota_cuestionario,
        feedback,
        datetime.utcnow().isoformat()
    ))
    conn.commit()
    conn.close()

def obtener_aggregado_cuestionario():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT 
            id_estudiante,
            nrc_curso,
            SUM(COALESCE(nota_cuestionario,0)) AS calificacion_cuestionario,
            COUNT(*) AS n_registros_cuestionario
        FROM respuestas_cuestionario
        GROUP BY id_estudiante, nrc_curso
    """, conn)
    conn.close()
    return df

def obtener_ultimo_taller():
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT tr.id_estudiante, tr.nrc_curso, tr.nota_taller AS calificacion_taller, tr.fecha_registro
        FROM taller_resultados tr
        INNER JOIN (
            SELECT id_estudiante, nrc_curso, MAX(fecha_registro) AS max_fecha
            FROM taller_resultados
            GROUP BY id_estudiante, nrc_curso
        ) ult
        ON tr.id_estudiante = ult.id_estudiante
        AND tr.nrc_curso = ult.nrc_curso
        AND tr.fecha_registro = ult.max_fecha
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def obtener_ultimo_cuestionario():
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT rc.id_estudiante, rc.nrc_curso, rc.nota_cuestionario AS calificacion_cuestionario, rc.fecha_registro
        FROM respuestas_cuestionario rc
        INNER JOIN (
            SELECT id_estudiante, nrc_curso, MAX(fecha_registro) AS max_fecha
            FROM respuestas_cuestionario
            GROUP BY id_estudiante, nrc_curso
        ) ult
        ON rc.id_estudiante = ult.id_estudiante
        AND rc.nrc_curso = ult.nrc_curso
        AND rc.fecha_registro = ult.max_fecha
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


# ----------------------------
# FUNCIONES DE EXTRACCIÓN Y VALIDACIÓN
# ----------------------------
def extraer_texto_y_urls(file_bytes):
    texto = ""
    urls = []
    with pdfplumber.open(BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            texto += page_text + "\n"
            encontrados = re.findall(r"https://www\.credly\.com/go/\S+", page_text)
            urls.extend(encontrados)
    return texto.strip(), list(set(urls))

def procesar_texto(texto):
    lineas = [l.strip() for l in texto.splitlines() if l.strip()]

    curso_idx = None
    for i, linea in enumerate(lineas):
        if re.match(r"^AWS", linea, re.IGNORECASE):
            curso_idx = i
            break

    if curso_idx is None or curso_idx + 2 >= len(lineas):
        return None, None, None, None

    curso = lineas[curso_idx]
    horas = lineas[curso_idx + 1]
    fecha = lineas[curso_idx + 2]
    nombre = " ".join(lineas[:curso_idx])

    return nombre, curso, horas, fecha

def validar_badge_publico(url):
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            return True, "✅ Badge válido"
        else:
            return False, f"❌ Badge no válido (status {resp.status_code})"
    except Exception as e:
        return False, f"❌ Error al validar: {e}"

# ----------------------------
# FUNCIONES DE EVALUACIÓN (Gemini)
# ----------------------------
import re

def evaluar_taller_redes_neuronales(contenido_estudiante: str):
    """
    Evalúa el código entregado por el estudiante para el Taller de Redes Neuronales.
    Devuelve (nota_taller, feedback) o (None, error_message)
    """
    prompt = f"""
Eres un profesor experto en Ciencia de Datos y Redes Neuronales en Python (Keras/TensorFlow).
Evalúa y califica el código enviado por un estudiante para un taller práctico de predicción de precios de vivienda con redes neuronales.

Criterios de evaluación:
1. Preprocesamiento de datos: EDA, imputación de valores nulos, manejo de outliers, escalado, división en entrenamiento y prueba.
2. Implementación del modelo: uso de Keras, modelo Secuencial, capas densas, activación ReLU en capas ocultas y capa de salida lineal.
3. Exploración de hiperparámetros: learning rate, número de capas y neuronas, epochs, batch size.
4. Optimización de hiperparámetros: función que pruebe combinaciones, evaluación con métricas (MSE, MAE).
5. Informe y conclusiones: análisis de resultados, hallazgos del preprocesamiento, sensibilidad a hiperparámetros.


Evalue solo lo que la plantilla pide en el notebook cargado, que es una plantilla. Y no quite puntos si hay mejores formas para realizar el ejercicio.
Evalue solo lo que se pide en lasa celdas del notebook. Evalue que si realiza todos los puntos,
si las celdas están vacias o no tiene el codigo requerido califique mal esos puntos.

Genera:
- Una calificación numérica entre 0 y 1.5.
- Retroalimentación clara, profesional y técnica sobre los puntos anteriores.

Código entregado por el estudiante:
{contenido_estudiante}
"""

    try:
        response = client.models.generate_content(
            model=gemini_model,
            contents=prompt
        )

        feedback = response.text

        # Extraer la primera aparición de un número decimal como nota
        match = re.search(r"(\d+(\.\d+)?)", feedback)
        if match:
            nota_taller = float(match.group(1))
            if nota_taller > 3.0:
                nota_taller = 3.0
        else:
            nota_taller = 1.5  # valor por defecto si no encuentra número

        return nota_taller, feedback

    except Exception as e:
        return None, f"Error en la llamada a la API: {e}"


def evaluar_respuestas_abiertas(respuestas_estudiante):
    """
    Evalúa un diccionario respuestas_estudiante: {1: 'texto', 2: 'texto', ...}
    Devuelve (nota_total_float, feedback_text_str, parsed_results_list_or_None)
    """
    import json, re

    # Respuestas esperadas para 15 preguntas
    respuestas_esperadas = {
    1: "tfds.load() permite cargar datasets estandarizados con particiones predefinidas, metadatos y preprocesamiento incluidos, lo que ahorra tiempo y asegura reproducibilidad frente a la carga manual.",
    2: "Consultar los metadatos permite conocer el tamaño de los conjuntos de train y test, lo cual es clave para dimensionar el entrenamiento, validar resultados y evitar sesgos.",
    3: "La normalización a valores entre 0 y 1 estabiliza el entrenamiento, mejora la convergencia y evita que los gradientes sean demasiado grandes; sin ella el modelo entrenaría más lento o de forma inestable.",
    4: "Sequential es simple y adecuado para arquitecturas lineales como MNIST; la API funcional se recomienda cuando existen múltiples entradas, salidas o arquitecturas más complejas.",
    5: "Flatten transforma la imagen 28x28 en un vector de 784 valores para conectarse con las capas densas; sin ella, la red no podría procesar correctamente los datos.",
    6: "relu ayuda a evitar el problema del gradiente desvanecido y acelera el aprendizaje; softmax convierte las salidas en probabilidades para clasificación multiclase.",
    7: "sparse_categorical_crossentropy se usa cuando las etiquetas son enteros; categorical_crossentropy requiere one-hot encoding. Ambas calculan la entropía cruzada, pero con diferente formato de etiquetas.",
    8: "Adam combina las ventajas de AdaGrad y RMSProp, ajustando la tasa de aprendizaje de forma adaptativa; suele converger más rápido que SGD en este tipo de problemas.",
    9: "Los parámetros batch_size y epochs influyen en la generalización: un batch muy pequeño aumenta ruido, uno muy grande puede sobreajustar; más epochs mejoran el ajuste pero pueden causar overfitting si no se controla con validación.",
    10: "model.evaluate() devuelve la pérdida y métricas como accuracy en el conjunto de test; accuracy indica el porcentaje de predicciones correctas sobre todas las muestras evaluadas.",
    11: "Guardar modelos con model.save() o pesos con save_weights() permite reanudar entrenamiento o hacer inferencia en producción sin volver a entrenar. Esto mantiene la consistencia entre entrenamiento y despliegue, y facilita reproducibilidad.",
    12: "EC2 permite provisionar instancias escalables y especializadas (GPU/CPU) para entrenamiento de ML, evitando inversión en hardware físico. Comparado con servidores locales, ofrece flexibilidad, escalabilidad y pago por uso, adaptándose a necesidades variables de cómputo.",
    13: "S3 permite almacenar grandes volúmenes de datos de forma duradera y escalable. Integrado con Keras, se puede cargar datasets en memoria o en streaming durante el entrenamiento, facilitando el manejo eficiente de datos masivos y la integración con pipelines de ML.",
    14: "Auto Scaling ajusta dinámicamente el número de instancias EC2 según la carga de trabajo. Esto es útil para entrenamientos intermitentes de redes neuronales grandes, evitando costos innecesarios en periodos de baja demanda y asegurando recursos suficientes en picos de trabajo.",
    15: "IaaS (EC2) ofrece infraestructura virtualizable, ideal para entrenar y experimentar modelos. PaaS (SageMaker) permite entrenar y desplegar modelos sin preocuparse por la infraestructura subyacente. SaaS ofrece soluciones listas para usar (como Rekognition), útil para consumir modelos sin desarrollar. La elección depende de si se busca control, flexibilidad o rapidez de despliegue."
    }

    n_preg = len(respuestas_esperadas)
    per_q = round(3.0 / n_preg, 2)  # cada pregunta vale 0.20 (3.0/15)

    # Construcción del prompt
    prompt = f"""Eres un profesor experto en Ciencia de Datos y AWS.
    Evalúa las siguientes {n_preg} respuestas de un estudiante comparándolas con la 'respuesta esperada'.
    El estudiante puede contestar con sus propias palabras, pero tu tarea es verificar si la respuesta del estudiante transmite la misma idea que la respuesta esperada.

    Instrucciones IMPORTANTES:
    Devuelve SOLO un JSON válido en este formato:
    {{
    "results": [
        {{"question": 1, "score": 0.20, "feedback": "Comentario corto"}},
        ...
    ],
    "total": 3.0
    }}

    Reglas:
    - "score" es un número decimal entre 0 y {per_q}, con dos decimales.
    - "total" es la suma de los scores, redondeada a 2 decimales y máximo 3.0.
    - No devuelvas texto fuera del JSON.
    - Si la respuesta del estudiante refleja correctamente el concepto central de la respuesta esperada, asigna 0.3 puntos.
    - Si la respuesta es parcialmente correcta (menciona algo relacionado pero incompleto o con errores menores), asigna 0.2 puntos.
    - Si la respuesta es fatla o es incorrecta o irrelevante, asigna 0.
    A continuación incluyo las respuestas esperadas y luego las respuestas del estudiantes
    """

    for k in range(1, n_preg+1):
        prompt += f"\nEsperada {k}: {respuestas_esperadas[k]}"
    for k in range(1, n_preg+1):
        resp = respuestas_estudiante.get(k, "").replace("\n", " ").strip()
        prompt += f"\nEstudiante {k}: {resp}"

    # Aquí iría la llamada a la API de evaluación (client.models.generate_content)
    # El resto del código que parsea JSON y construye feedback se mantiene igual
    # ...

    try:
        response = client.models.generate_content(
            model=gemini_model,
            contents=prompt
        )
        raw = response.text.strip()

        # Buscar JSON
        matches = re.findall(r"\{.*\}", raw, flags=re.DOTALL)
        if not matches:
            return None, f"(No se detectó JSON)\n\nRaw:\n{raw}", None

        json_text = matches[-1]
        data = json.loads(json_text)

        results = data.get("results", [])
        total = float(data.get("total", 0.0))

        if total > 3.0:
            total = 3.0

        # Feedback legible
        lines = []
        for r in results:
            q = r.get("question")
            s = float(r.get("score", 0.0))
            fb = r.get("feedback", "")
            lines.append(f"\n**Pregunta {q}:** {s:.2f}/{per_q}\n{fb}")
        lines.append(f"\nNota final: {total:.2f} / 3.00")

        return total, "\n".join(lines), results

    except Exception as e:
        return None, f"Error en la llamada a la API: {e}", None    


# ----------------------------
# INTERFAZ STREAMLIT
# ----------------------------
st.set_page_config(page_title="Validador de Badges + Taller", layout="wide")
# Título principal
st.title("Examen de IA y Ciencia de Datos")

# Subtítulo con derechos reservados
st.markdown("### Realizado por Alfredo Diaz  &nbsp;&nbsp;© UNB 2025")

# Imagen debajo
st.image(
    "https://apolo.unab.edu.co/files-asset/28174058/7-Alfredo-Antonio-Diaz-Claro.jpg?w=160&f=webp",
    caption="Alfredo Antonio Diaz Claro",
    use_column_width=True
)
init_db()

tabs = st.tabs([
    "📂 Cargar PDF",
    "📊 Registros en BD",
    "📝 Evaluar Taller",
    "📝 Evaluar Cuestionario",
    "📈 Calificaciones"
])

# ----------------------------
# Pestaña 1 - Subida de archivo
# ----------------------------
with tabs[0]:
    st.markdown("## Examen de Inteligencia Artificial 2025 UNAB")
    st.markdown(
    """
    📥 [Descargar examen (Examen_ID_NRC.ipynb)](https://raw.githubusercontent.com/adiacla/vivienda/refs/heads/main/Examen_Keras_Paso_a_Paso_plantilla.ipynb)
    """,  unsafe_allow_html=True)
    st.markdown("Sube cada PDF de uno en uno del badge obtenido en [Credly](https://www.credly.com/) tras completar el examen.")
        st.markdown("### Es obligatorio subir los badge para realizar el taller y el examen")
    id_input = st.text_input("ID del estudiante", key="input_id")
    nrc_input = st.text_input("NRC del curso", key="input_nrc")
    uploaded_file = st.file_uploader("Subir archivo PDF", type=["pdf"])

    if uploaded_file is not None and id_input and nrc_input:
        if st.button("Procesar y Guardar", key="procesar_guardar_pdf"):
            file_bytes = uploaded_file.read()
            texto, urls = extraer_texto_y_urls(file_bytes)
            urls_credly = [u for u in urls if u.startswith("https://www.credly.com/go/")]

            if not urls_credly:
                st.warning("⚠️ El PDF no tiene una URL de Credly para validar.")
            else:
                nombre, curso, horas, fecha = procesar_texto(texto)
                if not all([nombre, curso, horas, fecha]):
                    st.error("El PDF no tiene el formato esperado (nombre, curso, horas, fecha).")
                else:
                    for url in urls_credly:
                        valido, mensaje = validar_badge_publico(url)
                        guardar_en_db(id_input, nrc_input, nombre, curso, horas, fecha, url, valido, mensaje, None)
                        st.session_state["id_estudiante"] = id_input
                        st.session_state["nrc_curso"] = nrc_input
                        st.success(f"Guardado en BD: {id_input}, {nrc_input}, {nombre}, {curso}, {horas}, {fecha}, {url}")
                        st.info(mensaje)

# ----------------------------
# Pestaña 2 - Ver registros
# ----------------------------
with tabs[1]:
    st.header("Ver los Badges Cargados")

    df = obtener_registros()

    if df.empty:
        st.info("No hay registros guardados todavía.")
    else:
        # Filtrar por ID y NRC si se han digitado en la pestaña 1
        if st.session_state.get("input_id"):
            df = df[df["id"] == st.session_state["input_id"].strip()]

        if st.session_state.get("input_nrc"):
            df = df[df["nrc"] == st.session_state["input_nrc"].strip()]

        st.dataframe(df, width='stretch')
        
        
# ----------------------------
# Pestaña 3 - Evaluar Taller
# ----------------------------
with tabs[2]:
    st.header("Evaluar Taller de Redes Neuronales")

    archivo_taller = st.file_uploader(
        "Subir solución en Python (.py o .ipynb)", 
        type=["py", "ipynb"],
        key="uploader_taller"
    )

    comentarios_taller = st.text_area(
        "Comentarios / Observaciones (opcional)", 
        height=120, 
        key="comentarios_taller"
    )

    if st.button("Evaluar Taller", key="boton_evaluar_taller"):
        if archivo_taller is not None and "id_estudiante" in st.session_state and "nrc_curso" in st.session_state:
            try:
                nombre = archivo_taller.name.lower()

                # Leer contenido según tipo de archivo
                if nombre.endswith(".py"):
                    contenido_estudiante = archivo_taller.read().decode("utf-8", errors="ignore")

                elif nombre.endswith(".ipynb"):
                    nb_json = json.load(archivo_taller)
                    celdas = []
                    for celda in nb_json.get("cells", []):
                        if celda.get("cell_type") == "code":
                            celdas.append("".join(celda.get("source", [])))
                    contenido_estudiante = "\n".join(celdas)

                else:
                    st.error("⚠️ Formato de archivo no soportado.")
                    contenido_estudiante = None

                if contenido_estudiante:
                    # Evaluar el taller usando la nueva función para redes neuronales
                    nota_taller, feedback = evaluar_taller_redes_neuronales(contenido_estudiante)

                    if nota_taller is not None:
                        # Guardar resultado
                        guardar_resultado_taller(
                            st.session_state["id_estudiante"],
                            st.session_state["nrc_curso"],
                            nota_taller,
                            feedback,
                            comentarios_taller or ""
                        )
                        st.success(f"✅ Nota del taller guardada: {nota_taller} / 1.5")
                        st.session_state["nota_taller"] = nota_taller
                    else:
                        st.warning("⚠️ No se pudo calcular la nota automáticamente.")

                    st.subheader("Retroalimentación del taller")
                    st.write(feedback)

                else:
                    st.error("No se pudo extraer el contenido del archivo.")

            except Exception as e:
                st.error(f"❌ Error al procesar el archivo: {e}")

        else:
            st.error(
                "⚠️ Debes subir un archivo y haber cargado ID/NRC en la pestaña 'Cargar PDF'."
            )

# ----------------------------
# Pestaña 4 - Evaluar Cuestionario
# ----------------------------
with tabs[3]:
    st.header("Cuestionario de Preguntas Abiertas")

    preguntas_abiertas = [
        "¿Cuál es la diferencia entre cargar un dataset con tfds.load() y leer datos manualmente desde archivos locales, y qué ventajas ofrece tfds en proyectos de Deep Learning?",
        "¿Por qué es importante consultar los metadatos de un dataset con tfds, como el número de ejemplos de train y test, antes de entrenar un modelo?",
        "¿Por qué es necesario normalizar las imágenes de MNIST dividiendo sus valores de píxeles entre 255, y qué efecto tendría no hacerlo en el entrenamiento de la red neuronal?",
        "¿Qué ventajas tiene usar tf.keras.Sequential() para el taller de MNIST, y en qué casos sería más adecuado utilizar la API funcional de Keras?",
        "¿Qué rol cumple la capa Flatten() en este modelo y qué sucedería si se omitiera en la arquitectura propuesta?",
        "¿Por qué se utiliza relu en las capas ocultas y softmax en la capa de salida en este problema de clasificación multiclase?",
        "¿Por qué se utiliza sparse_categorical_crossentropy como función de pérdida en este taller y en qué se diferencia de categorical_crossentropy?",
        "¿Cuáles son las ventajas de usar el optimizador Adam frente a SGD en este taller y cómo afecta esto a la velocidad de convergencia del modelo?",
        "¿Qué parámetros dentro de model.fit() influyen directamente en la generalización del modelo y cómo se pueden ajustar para evitar overfitting?",
        "¿Qué información nos brinda el método model.evaluate() al final del entrenamiento y cómo se interpreta la métrica accuracy en el contexto del dataset MNIST?"
                
        # AWS Cloud Foundation
        "En AWS, ¿cuáles son las ventajas de usar EC2 para entrenar modelos de machine learning frente a usar servidores físicos locales?",
        "Explique cómo S3 puede integrarse con un pipeline de entrenamiento de redes neuronales para manejo eficiente de datasets grandes.",
        "¿Qué beneficios ofrece el Auto Scaling en AWS cuando se entrenan modelos que requieren grandes recursos computacionales de manera intermitente?",
        "Compare los modelos IaaS, PaaS y SaaS en AWS y explique cuál sería más adecuado para desplegar y consumir modelos de deep learning."
        ]
    respuestas_usuario = {}
    for i, pregunta in enumerate(preguntas_abiertas, start=1):
        st.markdown(f"**Pregunta {i}:** {pregunta}")
        respuesta = st.text_area(f"Respuesta {i}", key=f"resp_{i}", height=120)
        respuestas_usuario[i] = respuesta or ""

    # 👇 Aquí vuelve el botón
    if st.button("Evaluar Cuestionario", key="boton_evaluar_cuestionario"):
        if st.session_state.get("input_id") and st.session_state.get("input_nrc"):
            with st.spinner("Evaluando respuestas..."):
                nota_cuestionario, feedback_text, parsed = evaluar_respuestas_abiertas(respuestas_usuario)

            if nota_cuestionario is not None:
                st.success(f"✅ Nota cuestionario: {nota_cuestionario:.2f} / 3.0")

                # Guardar en base de datos con ID y NRC reales
                guardar_respuestas_cuestionario(
                    st.session_state["input_id"].strip(),
                    st.session_state["input_nrc"].strip(),
                    respuestas_usuario,
                    nota_cuestionario,
                    feedback_text
                )

                # También guardar en session_state
                st.session_state["nota_cuestionario"] = nota_cuestionario
            else:
                st.warning("⚠️ No se pudo obtener una nota numérica del modelo. Revisa el feedback.")

            st.subheader("Retroalimentación del cuestionario")
            st.markdown(feedback_text)
        else:
            st.error("⚠️ Debes haber ingresado ID y NRC en la pestaña 'Cargar PDF' antes de evaluar.")


# ----------------------------
# Pestaña 5 - CALIFICACIONES (resumen integrado)
# ----------------------------
with tabs[4]:
    st.header("📈 Resumen de Calificaciones")

    # Badges
    df_badges = obtener_calificaciones_badges()
    df_taller = obtener_ultimo_taller()
    df_cuestionario = obtener_ultimo_cuestionario()

    # Merge de todo
    df_merge = (
        df_badges
        .merge(df_taller, on=["id_estudiante", "nrc_curso"], how="outer")
        .merge(df_cuestionario, on=["id_estudiante", "nrc_curso"], how="outer")
    )

    # Rellenar NAs
    for col in ["calificacion_badge", "calificacion_taller", "calificacion_cuestionario"]:
        if col not in df_merge.columns:
            df_merge[col] = 0.0
        df_merge[col] = df_merge[col].fillna(0.0)

    # Nota total
    df_merge["nota_total"] = (
        df_merge["calificacion_badge"] +
        df_merge["calificacion_taller"] +
        df_merge["calificacion_cuestionario"]
    )

    # ----------------------------
    # 🔹 FILTRO por ID y NRC de la pestaña 1
    # ----------------------------
    id_filter = st.session_state.get("input_id", "").strip()
    nrc_filter = st.session_state.get("input_nrc", "").strip()

    if id_filter:
        df_merge = df_merge[df_merge["id_estudiante"] == id_filter]

    if nrc_filter:
        df_merge = df_merge[df_merge["nrc_curso"] == nrc_filter]

    # ----------------------------
    # Mostrar tabla resumen
    # ----------------------------
    if df_merge.empty:
        st.info("No hay calificaciones registradas aún para este estudiante y curso.")
    else:
        st.subheader("Resumen de calificaciones")
        st.dataframe(
            df_merge.reset_index(drop=True)[[
                "id_estudiante",
                "nrc_curso",
                "calificacion_badge",
                "calificacion_taller",
                "calificacion_cuestionario",
                "nota_total"
            ]],
            use_container_width=True
        )

        st.info("⚖️ Nota máxima posible = 5.0 (Badges 0.5 + Taller 1.5 + Cuestionario 3.0)")
        
            # ----------------------------
    # Botón para descargar la base de datos
    # ----------------------------
    with open(DB_PATH, "rb") as f:
        st.download_button(
            label="📥 Descargar base de datos SQLite",
            data=f,
            file_name="badges.db",
            mime="application/octet-stream"
        )


st.write("")  # espacio final




