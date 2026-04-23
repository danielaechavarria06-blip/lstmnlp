"""
Generador de Texto con LSTM
Aplicacion Streamlit - Curso Agentes de IA e Interfaces Multimodales
"""

import streamlit as st
import numpy as np
import json
import time
import os

st.set_page_config(
    page_title="Generador LSTM",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 🎨 ESTILOS CELESTE + BURBUJAS
st.markdown("""
<style>

/* 🌊 FONDO */
.stApp {
    background: linear-gradient(180deg, #dbeafe, #e0f2fe);
    color: #0f172a;
}

/* 🧠 TITULO */
.main-title {
    text-align: center;
    font-size: 2.5rem;
    font-weight: 800;
    color: #0284c7;
}

/* SUBTITULO */
.subtitle {
    text-align: center;
    color: #0369a1;
    font-size: 1rem;
}

/* TEXTO GENERADO */
.generated-text {
    background: linear-gradient(135deg, #f0f9ff, #bae6fd);
    border-radius: 15px;
    padding: 1.5rem;
    font-family: Georgia, serif;
    line-height: 1.8;
    border-left: 6px solid #0ea5e9;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: #e0f2fe;
}

/* BOTONES */
.stButton>button {
    background: linear-gradient(135deg, #38bdf8, #0ea5e9);
    color: white;
    border-radius: 12px;
    border: none;
    font-weight: bold;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.05);
    box-shadow: 0px 0px 15px rgba(14,165,233,0.5);
}

/* 🫧 BURBUJAS */
.bubble {
    position: fixed;
    border-radius: 50%;
    opacity: 0.2;
    background: #38bdf8;
    animation: float 12s infinite;
}

.b1 { width: 80px; height: 80px; left: 10%; bottom: -100px; animation-delay: 0s;}
.b2 { width: 60px; height: 60px; left: 30%; bottom: -100px; animation-delay: 3s;}
.b3 { width: 100px; height: 100px; left: 60%; bottom: -100px; animation-delay: 6s;}
.b4 { width: 50px; height: 50px; left: 80%; bottom: -100px; animation-delay: 9s;}

@keyframes float {
    0% { transform: translateY(0); }
    100% { transform: translateY(-120vh); }
}

</style>

<div class="bubble b1"></div>
<div class="bubble b2"></div>
<div class="bubble b3"></div>
<div class="bubble b4"></div>
""", unsafe_allow_html=True)


# ── Funciones ────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model_and_metadata(model_path, metadata_path):
    try:
        import tensorflow as tf
        from tensorflow import keras
        model = keras.models.load_model(model_path)
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        metadata["idx_to_char"] = {int(k): v for k, v in metadata["idx_to_char"].items()}
        return model, metadata, None
    except Exception as e:
        return None, None, str(e)


def is_embedding_model(model):
    first = model.layers[0]
    return hasattr(first, "input_dim") or first.__class__.__name__ == "Embedding"


def sample_temperature(preds, temperature=1.0):
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds + 1e-10) / temperature
    preds = np.exp(preds - np.max(preds))
    preds /= preds.sum()
    return np.argmax(np.random.multinomial(1, preds, 1))


def prepare_input(window, char_to_idx, vocab_size, use_embedding):
    indices = [char_to_idx.get(c, 0) for c in window]
    if use_embedding:
        return np.array([indices], dtype=np.int32)
    else:
        x = np.array(indices, dtype=np.float32) / float(vocab_size)
        return x.reshape(1, len(window), 1)


def generate_full_text(model, seed_text, char_to_idx, idx_to_char,
                        seq_length, vocab_size, n_chars=200, temperature=0.8):
    use_emb = is_embedding_model(model)
    seed_text = seed_text.lower()
    if len(seed_text) < seq_length:
        seed_text = seed_text.rjust(seq_length)
    seed_text = seed_text[-seq_length:]
    seed_text = "".join(c if c in char_to_idx else " " for c in seed_text)

    generated = ""
    window = list(seed_text)

    for _ in range(n_chars):
        x = prepare_input(window[-seq_length:], char_to_idx, vocab_size, use_emb)
        preds = model.predict(x, verbose=0)[0]
        next_char = idx_to_char[sample_temperature(preds, temperature)]
        generated += next_char
        window.append(next_char)

    return generated


# ── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Configuración")
    st.markdown("---")

    model_file    = st.file_uploader("Modelo (.keras o .h5)", type=["keras", "h5"])
    metadata_file = st.file_uploader("Metadatos (.json)", type=["json"])

    st.markdown("---")

    temperature = st.slider("🌡️ Temperatura", 0.1, 2.0, 0.8, 0.05)

    if temperature < 0.5:
        st.caption("🧊 Conservador")
    elif temperature < 1.0:
        st.caption("⚖️ Balanceado")
    elif temperature < 1.4:
        st.caption("🔥 Creativo")
    else:
        st.caption("🌪️ Muy loco")

    n_chars = st.slider("📏 Longitud", 50, 500, 200, 50)

    st.markdown("---")

    seeds = [
        "en un lugar de la mancha",
        "el caballero miro al horizonte",
        "sancho panza respondio",
        "con estas razones perdia",
        "el hidalgo tomo la espada",
    ]

    selected_seed = st.selectbox("✨ Semilla:", ["(personalizada)"] + seeds)


# ── MAIN ────────────────────────────────────────────────────────────────────

st.markdown('<h1 class="main-title">🧠 Generador LSTM</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Texto creativo con IA 💙</p>', unsafe_allow_html=True)
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["✨ Generar", "🌡️ Temperatura", "📚 Teoría"])


# ── TAB 1 ───────────────────────────────────────────────────────────────────

with tab1:
    col1, col2 = st.columns(2)

    with col1:
        default = selected_seed if selected_seed != "(personalizada)" else ""
        seed_input = st.text_area("✍️ Escribe tu texto:", value=default, height=100)

        gen_btn = st.button("✨ Generar Texto", use_container_width=True)

    with col2:
        output = st.empty()
        output.markdown('<div class="generated-text">Tu texto aparecerá aquí...</div>', unsafe_allow_html=True)

    if gen_btn:
        if not seed_input.strip():
            st.error("Escribe algo 😒")
        elif model_file and metadata_file:

            ext = "keras" if model_file.name.endswith(".keras") else "h5"

            with open(f"/tmp/model.{ext}", "wb") as f:
                f.write(model_file.read())

            with open("/tmp/metadata.json", "wb") as f:
                f.write(metadata_file.read())

            model, meta, err = load_model_and_metadata(f"/tmp/model.{ext}", "/tmp/metadata.json")

            if err:
                st.error(err)
            else:
                with st.spinner("Generando magia... ✨"):
                    texto = generate_full_text(
                        model, seed_input,
                        meta["char_to_idx"], meta["idx_to_char"],
                        meta["seq_length"], meta["vocab_size"],
                        n_chars, temperature
                    )

                output.markdown(f'<div class="generated-text">{texto}</div>', unsafe_allow_html=True)

        else:
            st.info("Modo demo activado ✨")
            output.markdown('<div class="generated-text">Ejemplo creativo generado...</div>', unsafe_allow_html=True)


# ── TAB 2 Y 3 (sin cambios funcionales) ──────────────────────────────────────

with tab2:
    st.write("Comparación de temperatura aquí...")

with tab3:
    st.write("Teoría aquí...")

st.markdown("---")
st.markdown("<div style='text-align:center; color:#0284c7;'>Hecho con amor 💙</div>", unsafe_allow_html=True)
