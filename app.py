
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

/* Logo nella barra */
.top-bar img {
    height: 3.5cm;
    object-fit: contain;
}

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
    color: #222222;
    font-size: 2.6rem;
    line-height: 1.2;
    font-weight: 700;
    margin-top: 1rem;
    margin-bottom: 0.5rem;
    text-shadow: 1px 1px 2px rgba(255,255,255,0.6);
}

.custom-title span {
    display: block;
    font-weight: 400; /* non grassetto */
    font-size: 2.2rem; /* stessa grandezza */
}
</style>

<div class="top-bar">
    <img src="Palazzo_Adriano-Stemma.png" alt="Logo Comune">
</div>
""", unsafe_allow_html=True)

# Layout colonne e logo centrato
col1, col2, col3 = st.columns([2, 3, 2])  # col2 più larga
with col2:
    st.image("newnew.png", width=500)

# Titolo sotto l'immagine
st.markdown("""
<h1 class="custom-title">
  Assistente Virtuale<br>
  <span>del Comune di Palazzo Adriano</span>
</h1>
""", unsafe_allow_html=True)

# ---- Lettura PDF e chatbot ----
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

# Footer fisso in basso
st.markdown("""
<div class="custom-footer" style="
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    text-align: center;
    font-size: 0.75rem;
    color: #666666;
    background: transparent;
    padding: 10px 0;
    z-index: 9999;">
    Questo assistente utilizza l’AI e potrebbe commettere errori. Verifica sempre le informazioni importanti. | © 2025 – Sviluppato da Emily D'Ugo
</div>
""", unsafe_allow_html=True)
