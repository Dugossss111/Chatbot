
import streamlit as st

chiave = st.secrets["superkey"]

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI

from PyPDF2 import PdfReader
from PIL import Image

# --- CSS principale ---
st.markdown("""
<style>
# Barra rossa con logo
.top-bar {
    background-color: #d71921;
    height: 4cm;
    width: 100%;
    position: fixed;
    top: 0;
    left: 0;
    z-index: 9999;
    display: flex;
    align-items: center;
    padding-left: 20px;
}
.top-bar img {
    height: 3.5cm;
    width: auto;
    object-fit: contain;
}
body {
    margin-top: 4.5cm; /* Spazio per la barra fissa */
}
</style>

<div class="top-bar">
    <img src="Palazzo_Adriano-Stemma.png" alt="Logo Comune">
</div>
""", unsafe_allow_html=True)

/* Spazio sotto barra */
[data-testid="stAppViewContainer"] {
    padding-top: 4.5rem;
    background-color: #f9f9f9;
}

/* Titolo personalizzato */
.custom-title {
    position: relative;
    z-index: 10000;
    text-align: center;
    color: #222222 !important;
    font-size: 2.6rem;
    line-height: 1.2;
    font-weight: 700;
    margin-top: 1rem;
    margin-bottom: 0.5rem;
}
.custom-title span {
    display: block;
    font-weight: 400;
    font-size: 2.2rem;
}

/* Container per titolo + input, spostato in alto */
#content-container {
    margin-top: -2cm;
}

/* Stile per campo input con bordo rosso sempre */
input[type="text"] {
    background-color: #f9f9f9 !important;
    color: #222222 !important;
    border: 2px solid #d71921 !important;
    border-radius: 6px !important;
    padding: 10px !important;
    font-size: 16px !important;
    outline: none !important;
    box-shadow: none !important;
    caret-color: #d71921 !important;
}
input[type="text"]:focus,
input[type="text"]:not(:focus) {
    border: 2px solid #d71921 !important;
    box-shadow: none !important;
    outline: none !important;
}

/* Placeholder visibile e ben leggibile */
input::placeholder {
    color: #222222 !important;
    opacity: 1 !important;
    font-weight: 600 !important;
    text-shadow: 0 0 2px white !important;
}

/* Footer fisso in basso */
.custom-footer {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    text-align: center;
    font-size: 0.75rem;
    color: #666666;
    background: transparent;
    padding: 10px 0;
    z-index: 9999;
}
</style>

<div class="top-bar">
    <img src="Palazzo_Adriano-Stemma.png" alt="Logo Comune">
</div>
""", unsafe_allow_html=True)

# Layout colonne e logo centrale
col1, col2, col3 = st.columns([2, 2.2, 2])
with col2:
    st.image("aaa.png", width=500)

# Contenitore con id per spostare titolo + input in alto
st.markdown('<div id="content-container">', unsafe_allow_html=True)

# Titolo
st.markdown("""
<div class="custom-title">
  Assistente Virtuale<br>
  <span>del Comune di Palazzo Adriano</span>
</div>
""", unsafe_allow_html=True)

# Input
domanda = st.text_input("Ciao, sono Pai. In cosa posso esserti utile?")

st.markdown('</div>', unsafe_allow_html=True)

# ---- Lettura PDF e chatbot ----
file = "Comune_di_Palazzo_Adriano.pdf"

if file is not None:
    testo_letto = PdfReader(file)

    testo = ""
    for pagina in testo_letto.pages:
        testo += pagina.extract_text() or ""

    testo_spezzato = RecursiveCharacterTextSplitter(
        separators=["\n", " ", ""],
        chunk_size=800,
        chunk_overlap=100,
        length_function=len
    )

    pezzi = testo_spezzato.split_text(testo)

    embeddings = OpenAIEmbeddings(openai_api_key=chiave)
    vector_store = FAISS.from_texts(pezzi, embeddings)

    if domanda:
        with st.spinner("Sto cercando le informazioni che mi hai richiesto..."):
            rilevanti = vector_store.similarity_search(domanda)

            llm = ChatOpenAI(
                openai_api_key=chiave,
                temperature=0.7,
                max_tokens=1000,
                model_name="gpt-3.5-turbo-0125"
            )

            chain = load_qa_chain(llm, chain_type="stuff")
            risposta = chain.run(input_documents=rilevanti, question=domanda)

        st.markdown(f"""
        <div style="
            background: #fff3f4;
            border-left: 4px solid #d71921;
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 0 8px 8px 0;
            color: #222222;
            font-size: 1.1rem;">
            {risposta.replace('\\n', '<br>')}
        </div>
        """, unsafe_allow_html=True)

# Footer fisso in basso
st.markdown("""
<div class="custom-footer">
    Questo assistente utilizza l’AI e potrebbe commettere errori. Verifica sempre le informazioni importanti. | © 2025 – Sviluppato da Emily D'Ugo
</div>
""", unsafe_allow_html=True)
