
import streamlit as st

chiave = st.secrets["superkey"]

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI

from PyPDF2 import PdfReader
from PIL import Image


# --- CSS e banner ---
st.markdown("""
<style>
.top-banner {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    z-index: 9999;
}

[data-testid="stAppViewContainer"] {
    padding-top: 320px;  /* Regola in base all’altezza di banner.png */
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

/* Campo input stile rosso */
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

/* Placeholder leggibile */
input::placeholder {
    color: #222222 !important;
    opacity: 1 !important;
    font-weight: 600 !important;
    text-shadow: 0 0 2px white !important;
}

/* Footer fisso */
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

/* Responsività mobile */
@media (max-width: 600px) {
    .top-banner img {
        max-height: 180px;
    }
    [data-testid="stAppViewContainer"] {
        padding-top: 200px;
    }
}
</style>

<!-- Nuovo banner immagine -->
<div class="top-banner">
    <img src="banner.png" style="width: 100%; height: auto;">
</div>
""", unsafe_allow_html=True)

# --- Titolo centrale ---
st.markdown("""
<div class="custom-title">
  Assistente Virtuale<br>
  <span>del Comune di Palazzo Adriano</span>
</div>
""", unsafe_allow_html=True)

# --- Campo input ---
domanda = st.text_input("Ciao, sono Pai. In cosa posso esserti utile?")

# --- Lettura PDF e risposta con AI ---
file = "Comune_di_Palazzo_Adriano.pdf"

if file:
    reader = PdfReader(file)
    testo = ""
    for pagina in reader.pages:
        testo += pagina.extract_text() or ""

    splitter = RecursiveCharacterTextSplitter(
        separators=["\n", " ", ""],
        chunk_size=800,
        chunk_overlap=100,
        length_function=len
    )
    pezzi = splitter.split_text(testo)

    embeddings = OpenAIEmbeddings(openai_api_key=chiave)
    vector_store = FAISS.from_texts(pezzi, embeddings)

    if domanda:
        with st.spinner("Sto cercando le informazioni che mi hai richiesto..."):
            risultati = vector_store.similarity_search(domanda)

            llm = ChatOpenAI(
                openai_api_key=chiave,
                temperature=0.7,
                max_tokens=1000,
                model_name="gpt-3.5-turbo-0125"
            )

            chain = load_qa_chain(llm, chain_type="stuff")
            risposta = chain.run(input_documents=risultati, question=domanda)

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

# --- Footer ---
st.markdown("""
<div class="custom-footer">
    Questo assistente utilizza l’AI e potrebbe commettere errori. Verifica sempre le informazioni importanti. | © 2025 – Sviluppato da Emily D'Ugo
</div>
""", unsafe_allow_html=True)
