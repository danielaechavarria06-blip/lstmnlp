"""
Generador de Texto con LSTM
"""

import streamlit as st
import numpy as np
import json
import os
from PIL import Image

st.set_page_config(
    page_title="Generador LSTM",
    page_icon="🧠",
    layout="wide"
)

# 🎨 ESTILOS CELESTE + BURBUJAS
st.markdown("""
<style>
.stApp {
    background: linear-gradient(180deg, #dbeafe, #e0f2fe);
}

/* TITULO */
.main-title {
    text-align: center;
    font-size: 2.5rem;
    font-weight: 800;
    color: #0284c7;
}

/* SUB */
.subtitle {
    text-align: center;
    color: #0369a1;
}

/* TEXTO */
.generated-text {
    background: linear-gradient(135deg, #f0f9ff, #bae6fd);
    border-radius: 15px;
    padding: 1.5rem;
    font-family: Georgia;
    border-left: 6px solid #0ea5e9;
}

/* BOTONES */
.stButton>button {
    background: linear-gradient(135deg, #38bdf8, #0ea5e9);
    color: white;
    border-radius: 12px;
    border: none;
    font-weight: bold;
}

/* BURBUJAS */
.bubble {
    position: fixed;
    border-radius: 50%;
    opacity: 0.2;
    background: #38bdf8;
    animation: float 12s infinite;
}

.b1 { width: 80px; height: 80px; left: 10%; bottom: -100px;}
.b2 { width: 60px; height: 60px; left: 30%; bottom: -100px;}
.b3 { width: 100px; height: 100px; left: 60%; bottom: -100px;}
.b4 { width: 50px; height: 50px; left: 80%; bottom: -100px;}

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

# 🐋 HEADER CON IMAGEN
colA, colB = st.columns([1,3])

with colA:
    bot = Image.open("bot.jpg")  # 👈 tu imagen aquí
    st.image(bot, width=120)

with colB:
    st.markdown('<h1 class="main-title">🧠 Generador LSTM</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Tu bot creativo del mar 💙</p>', unsafe_allow_html=True)

st.markdown("---")

# ── FUNCIONES ─────────────────────────

@st.cache_resource
def load_model_and_metadata(model_path, metadata_path):
    try:
        from tensorflow import keras
        model = keras.models.load_model(model_path)
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        metadata["idx_to_char"] = {int(k): v for k, v in metadata["idx_to_char"].items()}
        return model, metadata, None
    except Exception as e:
        return None, None, str(e)

def sample_temperature(preds, temperature=1.0):
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds + 1e-10) / temperature
    preds = np.exp(preds - np.max(preds))
    preds /= preds.sum()
    return np.argmax(np.random.multinomial(1, preds, 1))

def generate_full_text(model, seed_text, char_to_idx, idx_to_char,
                      seq_length, vocab_size, n_chars=200, temperature=0.8):

    seed_text = seed_text.lower()
    seed_text = seed_text[-seq_length:]

    generated = ""
    window = list(seed_text)

    for _ in range(n_chars):
        x = np.array([char_to_idx.get(c, 0) for c in window]).reshape(1, seq_length, 1)
        preds = model.predict(x, verbose=0)[0]
        next_char = idx_to_char[sample_temperature(preds, temperature)]
        generated += next_char
        window.append(next_char)

    return generated


# ── SIDEBAR ─────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Configuración")

    model_file = st.file_uploader("Modelo", type=["keras", "h5"])
    metadata_file = st.file_uploader("Metadatos", type=["json"])

    temperature = st.slider("🌡️ Temperatura", 0.1, 2.0, 0.8)
    n_chars = st.slider("📏 Longitud", 50, 500, 200)


# ── MAIN ─────────────────────────

col1, col2 = st.columns(2)

with col1:
    seed_input = st.text_area("✍️ Escribe tu texto:")

    gen_btn = st.button("✨ Generar Texto", use_container_width=True)

with col2:
    output = st.empty()
    output.markdown('<div class="generated-text">Tu texto aparecerá aquí...</div>', unsafe_allow_html=True)

if gen_btn:
    if not seed_input.strip():
        st.error("Escribe algo 😒")

    elif model_file and metadata_file:

        with open("/tmp/model.h5", "wb") as f:
            f.write(model_file.read())

        with open("/tmp/meta.json", "wb") as f:
            f.write(metadata_file.read())

        model, meta, err = load_model_and_metadata("/tmp/model.h5", "/tmp/meta.json")

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
        st.info("Modo demo ✨")
        output.markdown('<div class="generated-text">Había una vez un bot del océano que escribía historias mágicas...</div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown("<div style='text-align:center;'>Hecho con amor 💙</div>", unsafe_allow_html=True)
