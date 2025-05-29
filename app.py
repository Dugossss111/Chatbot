
import streamlit as st

chiave = st.secrets["superkey"]

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI

from PyPDF2 import PdfReader
from PIL import Image

# --- CSS principale + titolo ---
st.markdown("""
<style>
/* Barra rossa in alto */
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

/* Logo in alto a sinistra nella barra */
.top-bar img {
    height: 3.5cm;
    object-fit: contain;
}

/* Spazio sotto la barra */
[data-testid="stAppViewContainer"] {
    padding-top: 4.5rem;
    background-color: #f9f9f9;
}

/* Titolo principale */
.custom-title {
    position: relative;
    z-index: 10000;
    text-align: center;
    color: #222222;
    font-weight: 700;
    font-size: 2.6rem;
    text-shadow: 1px 1px 2px rgba(255,255,255,0.6);
    line-height: 1.2;
    margin-top: 0;
    margin-bottom: 0.2rem;
}

.custom-title span:last-child {
    font-size: 2.6rem;
    font-weight: 400;
    display: block;
    line-height: 1.2;
}

/* Colore del testo dell'output */
html, body, [data-testid="stMarkdownContainer"] {
    color: #222222 !important;
}

/* Input testo personalizzato */
.stTextInput > div > div > input {
    border: 2px solid #d71921;
    border-radius: 10px;
    padding: 10px 15px;
    font-size: 16px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
}

/* Focus input */
div.stTextInput > div > input:focus {
    border-bottom: 1px solid #666666 !important;
    outline: none !important;
    box-shadow: none !important;
}

/* Placeholder */
div.stTextInput > div > input::placeholder {
    color: #222222 !important;
    opacity: 1 !important;
}

/* Label input */
div.stTextInput > label {
    color: #222222;
    font-weight: 500;
    font-size: 1rem;
    margin-bottom: 4px;
    display: block;
    opacity: 0.7;
    transition: opacity 0.3s ease;
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

<!-- Barra e logo -->
<div class="top-bar">
    <img src="Palazzo_Adriano-Stemma.png" alt="Logo Comune">
</div>
""", unsafe_allow_html=True)

# Titolo su due righe (stessa grandezza, seconda riga non grassetto)
st.markdown("""
<h1 class="custom-title">
  Assistente Virtuale<br>
  <span>del Comune di Palazzo Adriano</span>
</h1>
""", unsafe_allow_html=True)

# Layout colonne con logo centrale
col1, col2, col3 = st.columns([2.4, 1.2, 2.4])
with col2:
    st.image("noname.png", width=600)

# Elaborazione PDF e chatbot
file = "Comune_di_Palazzo_Adriano.pdf"

if file is not None:
    testo_letto = PdfReader(file)

    testo = ""
    for pagina in testo_letto.pages:
        testo += pagina.extract_text()

    testo_spezzato = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )

    pezzi = testo_spezzato.split_text(testo)

    embeddings = OpenAIEmbeddings(openai_api_key=chiave)
    vector_store = FAISS.from_texts(pezzi, embeddings)

    domanda = st.text_input("Digita qui la tua richiesta:")

    if domanda:
        with st.spinner("Sto cercando le informazioni che mi hai richiesto..."):
            rilevanti = vector_store.similarity_search(domanda)

            llm = ChatOpenAI(
                openai_api_key=chiave,
                temperature=1.0,
                max_tokens=1000,
                model_name="gpt-3.5-turbo-0125"
            )

            chain = load_qa_chain(llm, chain_type="stuff")
            risposta = chain.run(input_documents=rilevanti, question=domanda)

        st.write(risposta)

# Footer
st.markdown("""
<div class="custom-footer">
    Questo assistente utilizza l’AI e potrebbe commettere errori. Verifica sempre le informazioni importanti. | © 2025 – Sviluppato da Emily D'Ugo
</div>
""", unsafe_allow_html=True)
