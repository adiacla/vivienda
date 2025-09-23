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
def evaluar_taller(contenido_estudiante: str):
    """
    Evalúa el código enviado por el estudiante.
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

Verifique el el archivo cargada tiene celdas donde se realiza cada paso. Si las celdas estan comentadas, no las considereo si estan sin codigo tampoco.
Igualemente si el codigo no corre o tiene errores, evalue el codigo y no la ejecucion.
Ademas revise que el codigo siga las mejores practicas de Python y Keras y tiene la lógica correcta.
Código entregado por el estudiante:
{contenido_estudiante}

Genera una calificación numérica entre 0 y 1.5 y retroalimentación profesional clara y técnica.
"""
    try:
        response = client.models.generate_content(
            model=gemini_model,
            contents=prompt
        )

        feedback = response.text

        # Buscar la primera aparición de un número con posible decimal
        match = re.search(r"(\d+(\.\d+)?)", feedback)
        if match:
            nota_taller = float(match.group(1))
            if nota_taller > 1.5:
                nota_taller = 1.5
        else:
            nota_taller = 1.0  # valor por defecto si no encuentra número

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
        1: "Un modelo secuencial en Keras se define usando keras.Sequential(), agregando capa por capa. Es útil para redes simples donde cada capa tiene un único flujo de datos. Ventajas: fácil de implementar y depurar. Limitaciones: no permite múltiples entradas o salidas ni arquitecturas complejas con saltos entre capas, lo cual requiere el API funcional.",
        2: "La función de activación introduce no linealidad al perceptrón, permitiendo al modelo aprender relaciones complejas. Activaciones inapropiadas pueden causar saturación (sigmoid/tanh) o gradientes nulos, afectando la convergencia y la velocidad de entrenamiento.",
        3: "En clasificación binaria se suele usar 'sigmoid' como activación en la capa de salida y 'binary_crossentropy' como función de pérdida. En clasificación multiclase se usa 'softmax' con 'categorical_crossentropy'. Elegir incorrectamente puede dar predicciones inconsistentes y afectar la optimización.",
        4: "En regresión lineal se usa típicamente activación lineal (None) en la salida y 'mean_squared_error' o 'mean_absolute_error' como función de pérdida. Esto permite que la red prediga cualquier valor continuo sin limitarlo a un rango específico.",
        5: "El número de capas densas y neuronas se ajusta para balancear capacidad del modelo y riesgo de overfitting. Pocas capas/neuronas → underfitting, demasiadas → overfitting. Se recomienda empezar con pocas capas y aumentar según desempeño en validación, usando regularización si es necesario.",
        6: "El batch size afecta el cálculo de gradientes: batches pequeños generan entrenamiento más ruidoso pero con mejor generalización, batches grandes entrenan más rápido pero pueden estancarse en mínimos locales. El número de epochs controla cuántas veces se recorre todo el dataset; demasiados epochs pueden overfit y pocos underfit.",
        7: "SGD es básico y puede ser lento; Adam combina momentum y adaptación de learning rate, acelerando la convergencia y estabilidad; RMSprop adapta learning rate para cada parámetro, útil en problemas con gradientes ruidosos. La elección afecta rapidez y estabilidad del entrenamiento.",
        8: "Accuracy es útil para problemas balanceados, pero en datasets desbalanceados métricas como precision, recall y F1-score son más informativas. AUC mide capacidad de discriminación. Elegir la métrica adecuada permite evaluar correctamente el desempeño según el tipo de problema.",
        9: "La inicialización de pesos evita que los gradientes se saturen o se vuelvan demasiado pequeños/grandes. Estrategias comunes: He (para ReLU) y Glorot/Xavier (para sigmoide/tanh). Una buena inicialización mejora la velocidad de convergencia y evita que el modelo quede atrapado en mínimos subóptimos.",
        10: "El método fit entrena el modelo pasando los datos de entrada y salida. Parámetros críticos: batch_size (número de muestras por paso), epochs (cuántas veces recorrer todo el dataset), validation_split o validation_data, callbacks y shuffle. Ajustar estos parámetros impacta la eficiencia y la generalización.",
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
    A continuación incluyo las respuestas esperadas y luego las respuestas del estudiante.
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
    📥 [Descargar examen (Examen_ID_NRC.ipynb)](https://raw.githubusercontent.com/adiacla/vivienda/refs/heads/main/Taller_Evaluativo_Red_neuronal_vivienda.ipynb)
    """,  unsafe_allow_html=True)
    st.markdown("Sube el PDF del badge obtenido en [Credly](https://www.credly.com/) tras completar el examen.")
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
    st.header("Registros en la Base de Datos")

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
                        st.success(f"✅ Nota del taller guardada: {nota_taller} / 3.0")
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
        # Redes neuronales con Keras
        "¿Cómo se define un modelo secuencial en Keras y cuáles son sus ventajas y limitaciones frente a modelos funcionales?",
        "En un perceptrón multicapa, ¿por qué es importante elegir correctamente la función de activación y cómo afecta a la convergencia del modelo?",
        "¿Cuál es la diferencia entre clasificación binaria y multiclase en Keras, y qué función de pérdida y activación se recomienda para cada caso?",
        "En un modelo de regresión lineal implementado con Keras, ¿qué función de activación y pérdida se utilizan, y por qué?",
        "¿Cómo se determina el número óptimo de capas densas y neuronas por capa en un modelo Keras para evitar underfitting y overfitting?",
        "Explique cómo la selección del tamaño del batch y del número de epochs influye en el entrenamiento y la generalización de un modelo Keras.",
        "¿Cuál es la diferencia entre optimizadores como SGD, Adam y RMSprop, y cómo influye su elección en la velocidad y estabilidad del entrenamiento?",
        "¿Cómo se utilizan métricas como accuracy, precision, recall y AUC en Keras, y por qué algunas métricas son más adecuadas según el tipo de problema?",
        "¿Qué efectos tiene la inicialización de pesos en redes neuronales y cuáles son las estrategias más comunes para mejorar la convergencia?",
        "Explique cómo funciona el método fit en Keras y qué parámetros son críticos para controlar el entrenamiento de un modelo.",
        "¿Por qué es importante guardar modelos, pesos y optimizadores en Keras, y cómo se hace de forma que permita reanudar el entrenamiento o hacer inferencia en producción?",
        
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


