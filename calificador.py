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
Eres un profesor experto en ciencia de datos y preprocesamiento de datos en Python.
Evalúa y califica el código enviado por un estudiante para un taller práctico de preprocesamiento de datos de vivienda.
... (omitido por brevedad: usa el prompt original que tenías) ...

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
    # (mantenemos tu prompt y lógica original; aquí se reutiliza)
    respuestas_esperadas = {
        1: "Porque los algoritmos de ML no aceptan valores nulos. Ignorarlos puede hacer que fallen o den resultados sesgados. En el taller de preprocesamiento de vivienda vimos que imputar valores nulos evita pérdida de información y asegura que las variables puedan ser usadas en modelos. Esto se hace en la etapa inicial de limpieza de datos.",
        2: "Porque las variables discretas/categóricas (como número de habitaciones) representan conteos enteros. Usar la moda conserva un valor válido (ej. 3 habitaciones), mientras que la media puede dar un número irreal (ej. 3.7). Esto se hace en la imputación de datos faltantes, garantizando que el dataset tenga valores consistentes.",
        3: "Se eliminan duplicados para que ciertos registros no tengan más peso del que deberían en el entrenamiento. En el taller vimos que mantener duplicados puede sesgar el modelo y reducir la diversidad de los datos. Esto se hace en la fase de limpieza estructural del dataset.",
        4: "Los algoritmos de ML solo aceptan entradas numéricas. Los encoders convierten categorías en números para que puedan ser procesadas en la etapa de transformación de variables. En el taller aplicamos OneHotEncoder para variables como el tipo de vivienda.",
        5: "Se usa OrdinalEncoder cuando la variable categórica tiene un orden natural (ej. calidad baja < media < alta). OneHotEncoder perdería esa relación. Esto se aplica en la etapa de codificación según el tipo de variable.",
        6: "Los outliers distorsionan medidas estadísticas como media y desviación estándar, y pueden desajustar los modelos haciendo que generalicen mal. En el taller los detectamos con boxplots antes de entrenar, dentro de la fase de análisis exploratorio y limpieza.",
        7: "StandardScaler transforma los datos a media 0 y desviación estándar 1, útil para algoritmos sensibles a varianza (regresión, SVM). MinMaxScaler escala al rango [0,1], ideal para redes neuronales o KNN. Esto se hace en la etapa de normalización/estandarización.",
        8: "El scaler debe ajustarse (fit) solo con datos de entrenamiento para evitar fuga de información. Si se ajusta con todo el dataset, se estaría usando información del test, contaminando la evaluación. Esto se aplica en la etapa de preparación de datos para modelado.",
        9: "Guardar encoders y scalers con joblib permite reutilizarlos con datos nuevos en producción. Esto garantiza consistencia entre entrenamiento y predicción. En el taller vimos que se deben guardar los objetos entrenados junto al modelo.",
        10: "El modelo de pago por uso de AWS evita inversiones iniciales en infraestructura. Permite escalar recursos de forma flexible según el preprocesamiento o entrenamiento que se necesite, en lugar de comprar servidores físicos costosos.",
        11: "La escalabilidad automática en AWS ajusta recursos según la carga, evitando caídas o costos innecesarios. A diferencia de un servidor físico, permite entrenar modelos grandes en picos de trabajo y liberar recursos después.",
        12: "IaaS ofrece infraestructura (máquinas virtuales), PaaS plataformas para desarrollo, y SaaS aplicaciones listas para usar. Comprenderlos es clave para elegir la opción adecuada según la etapa del pipeline: IaaS para cómputo flexible, PaaS para desplegar modelos, y SaaS para consumir soluciones ya listas."
    }

    n_preg = len(respuestas_esperadas)
    per_q = round(3.5 / n_preg, 4)

    prompt = f"""Eres un profesor experto en Ciencia de Datos y AWS.
        Evalúa las siguientes {n_preg} respuestas de un estudiante comparándolas con la 'respuesta esperada'.

        Instrucciones IMPORTANTES:
        Devuelve SOLO un JSON válido en este formato:
        {{
        "results": [
            {{"question": 1, "score": 0.25, "feedback": "Comentario corto"}},
            {{"question": 2, "score": 0.29, "feedback": "Comentario corto"}}
        ],
        "total": 0.54
        }}

        Reglas:
        - "score" es un número decimal entre 0 y {per_q}, con dos decimales.
        - "total" es la suma de los scores, redondeada a 2 decimales y máximo 3.50.
        - No devuelvas texto fuera del JSON.
        A continuación incluyo las respuestas esperadas y luego las respuestas del estudiante.

        Respuestas esperadas (por pregunta):
        """
    # Añadir respuestas esperadas
    for k in range(1, n_preg+1):
        prompt += f"\nEsperada {k}: {respuestas_esperadas[k]}"
    for k in range(1, n_preg+1):
        resp = respuestas_estudiante.get(k, "").replace("\n", " ").strip()
        prompt += f"\nEstudiante {k}: {resp}"

    try:
        response = client.models.generate_content(
            model=gemini_model,
            contents=prompt
        )
        raw = response.text.strip()

        # Buscar el último JSON válido en la respuesta
        matches = re.findall(r"\{.*\}", raw, flags=re.DOTALL)
        if not matches:
            return None, f"(No se detectó JSON)\n\nRaw:\n{raw}", None

        json_text = matches[-1]
        try:
            data = json.loads(json_text)
        except Exception:
            return None, f"(JSON inválido)\n\nRaw:\n{raw}", None

        results = data.get("results", [])
        total = data.get("total", None)

        if total is None:
            return None, f"(No se encontró 'total' en JSON)\n\nRaw:\n{raw}", results

        total = float(round(float(total), 2))
        if total > 3.5:
            total = 3.5

        # Construir feedback legible
        lines = []
        for r in results:
            q = r.get("question")
            s = float(r.get("score", 0.0))
            fb = r.get("feedback", "")
            lines.append(f"\n**Pregunta {q}:** {s:.2f}/{per_q}\n{fb}")
        lines.append(f"\nNota final: {total:.2f} / 3.50")

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
    📥 [Descargar examen (Examen_ID_NRC.ipynb)](https://raw.githubusercontent.com/adiacla/vivienda/refs/heads/main/Examen_ID_NRC.ipynb)
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
        st.dataframe(df, width='stretch')

# ----------------------------
# Pestaña 3 - Evaluar Taller
# ----------------------------
with tabs[2]:
    st.header("Evaluar Taller de Preprocesamiento de Datos")

    archivo_taller = st.file_uploader(
        "Subir solución en Python (.py o .ipynb)", 
        type=["py", "ipynb"],
        key="uploader_taller"
    )

    comentarios_taller = st.text_area("Comentarios / Observaciones (opcional)", height=120, key="comentarios_taller")

    if st.button("Evaluar Taller", key="boton_evaluar_taller"):
        if archivo_taller is not None and "id_estudiante" in st.session_state and "nrc_curso" in st.session_state:
            try:
                nombre = archivo_taller.name.lower()

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
                    nota_taller, feedback = evaluar_taller(contenido_estudiante)
                    if nota_taller is not None:
                        # Guardar en tabla taller_resultados (no en badges)
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
            st.error("⚠️ Debes subir un archivo y haber cargado ID/NRC en la pestaña 'Cargar PDF'.")

# ----------------------------
# Pestaña 4 - Evaluar Cuestionario
# ----------------------------
with tabs[3]:
    st.header("Cuestionario de Preguntas Abiertas")

    preguntas_abiertas = [
        "¿Por qué en el preprocesamiento es importante manejar los valores nulos y qué consecuencias tendría ignorarlos?",
        "¿Por qué en variables discretas (como número de habitaciones o baños) es más apropiado usar la moda para imputar valores faltantes en lugar de la media?",
        "¿Por qué se eliminan duplicados en un dataset antes de entrenar un modelo?",
        "¿Por qué se utilizan encoders para transformar variables categóricas en valores numéricos?",
        "¿En qué casos se debe usar un OrdinalEncoder en lugar de un OneHotEncoder?",
        "¿Por qué se detectan y eliminan outliers antes del entrenamiento de modelos?",
        "¿Cuál es la diferencia entre usar StandardScaler y MinMaxScaler y en qué escenarios se usan?",
        "¿Por qué se recomienda ajustar el scaler solo con los datos de entrenamiento?",
        "¿Por qué es importante guardar con joblib los transformadores como encoders o scalers?",
        "¿Por qué el modelo de pago por uso en AWS es ventajoso frente a la compra de infraestructura tradicional?",
        "¿Cuál es la ventaja de la escalabilidad automática (Auto Scaling) en la nube frente a un servidor físico?",
        "¿Cuál es la diferencia entre IaaS, PaaS y SaaS en AWS y por qué es importante comprender estos modelos?"
    ]

    respuestas_usuario = {}
    for i, pregunta in enumerate(preguntas_abiertas, start=1):
        st.markdown(f"**Pregunta {i}:** {pregunta}")
        respuesta = st.text_area(f"Respuesta {i}", key=f"resp_{i}", height=120)
        respuestas_usuario[i] = respuesta or ""

    if st.button("Evaluar Cuestionario", key="boton_evaluar_cuestionario"):
        if "id_estudiante" in st.session_state and "nrc_curso" in st.session_state:
            with st.spinner("Evaluando respuestas..."):
                nota_cuestionario, feedback_text, parsed = evaluar_respuestas_abiertas(respuestas_usuario)

            # Mostrar EXACTAMENTE la nota devuelta por Gemini (si existe)
            if nota_cuestionario is not None:
                st.success(f"✅ Nota cuestionario: {nota_cuestionario:.2f} / 3.5")
                # Guardar en base de datos
                guardar_respuestas_cuestionario(
                    st.session_state["id_estudiante"],
                    st.session_state["nrc_curso"],
                    respuestas_usuario,
                    nota_cuestionario,
                    feedback_text
                )
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
    st.header("Calificaciones consolidadas por estudiante y curso (NRC)")

    # Obtener agregados de cada tabla
    df_badges = obtener_calificaciones_badges()  # id_estudiante, nrc_curso, calificacion_badge
    df_taller = obtener_ultimo_taller()      # id_estudiante, nrc_curso, calificacion_taller
    df_cuestion = obtener_ultimo_cuestionario() # id_estudiante, nrc_curso, calificacion_cuestionario

    # Normalizar nombres de columnas para merges
    if not df_badges.empty:
        df_badges = df_badges.rename(columns={"id_estudiante": "id_estudiante", "nrc_curso": "nrc_curso"})
    else:
        df_badges = pd.DataFrame(columns=["id_estudiante", "nrc_curso", "calificacion_badge"])

    if not df_taller.empty:
        df_taller = df_taller.rename(columns={"id_estudiante": "id_estudiante", "nrc_curso": "nrc_curso"})
    else:
        df_taller = pd.DataFrame(columns=["id_estudiante", "nrc_curso", "calificacion_taller", "n_registros_taller"])

    if not df_cuestion.empty:
        df_cuestion = df_cuestion.rename(columns={"id_estudiante": "id_estudiante", "nrc_curso": "nrc_curso"})
    else:
        df_cuestion = pd.DataFrame(columns=["id_estudiante", "nrc_curso", "calificacion_cuestionario", "n_registros_cuestionario"])

    # Merge outer para no perder registros
    df_merge = pd.merge(df_badges, df_taller, on=["id_estudiante", "nrc_curso"], how="outer")
    df_merge = pd.merge(df_merge, df_cuestion, on=["id_estudiante", "nrc_curso"], how="outer")

    # Rellenar NAs con 0
    fill_cols = ["calificacion_badge", "calificacion_taller", "calificacion_cuestionario"]
    for c in fill_cols:
        if c not in df_merge.columns:
            df_merge[c] = 0.0
    df_merge[["calificacion_badge", "calificacion_taller", "calificacion_cuestionario"]] = df_merge[["calificacion_badge", "calificacion_taller", "calificacion_cuestionario"]].fillna(0.0)

    # Nota total = suma de los componentes (badge + taller + cuestionario)
    df_merge["nota_total"] = df_merge["calificacion_badge"] + df_merge["calificacion_taller"] + df_merge["calificacion_cuestionario"]

    if df_merge.empty:
        st.info("No hay calificaciones registradas aún en ninguna de las tablas.")
    else:
        # Filtro por id_estudiante
        estudiantes = sorted(df_merge["id_estudiante"].dropna().unique().tolist())
        estudiante_sel = st.selectbox("Filtrar por ID de estudiante (opcional):", ["Todos"] + estudiantes)

        df_display = df_merge.copy()
        if estudiante_sel != "Todos":
            df_display = df_display[df_display["id_estudiante"] == estudiante_sel]

        st.subheader("Resumen")
        st.dataframe(df_display.reset_index(drop=True), width='stretch')

        # Mostrar detalle: últimas entradas de taller y cuestionario para el estudiante seleccionado (si aplica)
        if estudiante_sel != "Todos":
            st.subheader("Detalle por tabla")

            # Detalle badges
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("""
                SELECT url, id, nrc, nombre, curso, horas, fecha, valido, mensaje, nota_taller
                FROM badges
                WHERE id = ? AND nrc = ?
            """, (estudiante_sel, st.session_state.get("nrc_curso", "")))
            rows_badges = c.fetchall()
            df_badges_det = pd.DataFrame(rows_badges, columns=["url", "id", "nrc", "nombre", "curso", "horas", "fecha", "valido", "mensaje", "nota_taller"])
            st.write("Badges registrados (filtrados por id y la última nrc registrada en session_state si aplica):")
            if df_badges_det.empty:
                st.write("No hay registros de badges para este estudiante con esa NRC (o no hay NRC en session).")
            else:
                st.dataframe(df_badges_det, width='stretch')

            # Detalle taller_resultados
            df_taller_det = pd.read_sql_query("""
                SELECT id_estudiante, nrc_curso, nota_taller, feedback, comentarios, fecha_registro
                FROM taller_resultados
                WHERE id_estudiante = ?
                ORDER BY fecha_registro DESC
            """, conn, params=(estudiante_sel,))
            st.write("Registros de Taller (más recientes primero):")
            if df_taller_det.empty:
                st.write("No hay resultados de taller para este estudiante.")
            else:
                st.dataframe(df_taller_det, width='stretch')

            # Detalle cuestionario
            df_cuestion_det = pd.read_sql_query("""
                SELECT id_estudiante, nrc_curso, nota_cuestionario, feedback, fecha_registro
                FROM respuestas_cuestionario
                WHERE id_estudiante = ?
                ORDER BY fecha_registro DESC
            """, conn, params=(estudiante_sel,))
            st.write("Registros de Cuestionario (más recientes primero):")
            if df_cuestion_det.empty:
                st.write("No hay registros de cuestionario para este estudiante.")
            else:
                st.dataframe(df_cuestion_det, width='stretch')

            conn.close()

st.write("")  # espacio final

