
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
/* --- BARRA ROSSA ALTA CON LOGO A SINISTRA --- */
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
    object-fit: contain;
}

/* --- SPAZIO SOTTO LA BARRA ROSSA --- */
[data-testid="stAppViewContainer"] {
    padding-top: 4.5rem;
    background-color: #f9f9f9;
}

/* --- TITOLO PERSONALIZZATO: DUE RIGHE COMPATTE --- */
.custom-title {
    text-align: center;
    position: relative;
    z-index: 10000;
    margin: 0;
}
.custom-title span:first-child {
    font-size: 2.6rem;
    font-weight: 700;
    color: #222222;
    line-height: 1;
}
.custom-title span:last-child {
    font-size: 1.5rem;
    font-weight: 400;
    color: #222222;
    line-height: 1;
}

/* --- INPUT PERSONALIZZATO --- */
.stTextInput > div > div > input {
    border: 2px solid #d71921;
    border-radius: 10px;
    padding: 10px 15px;
    font-size: 16px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
}
.stTextInput > div > input:focus {
    border-bottom: 1px solid #666666 !important;
    outline: none !important;
    box-shadow: none !important;
}
.stTextInput > div > input::placeholder {
    color: #222222 !important;
    opacity: 1 !important;
}
.stTextInput > label {
    color: #222222;
    font-weight: 500;
    font-size: 1rem;
    margin-bottom: 4px;
    display: block;
    opacity: 0.7;
    transition: opacity 0.3s ease;
}

/* --- FOOTER FISSO IN BASSO --- */
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

# --- LOGO CENTRALE SOTTO LA BARRA ---
col1, col2, col3 = st.columns([2.4, 1.2, 2.4])
with col2:
    st.image("noname.png", width=600)

# --- TITOLO COMPATTO A DUE RIGHE ---
st.markdown("""
<div class="custom-title">
  <span>Assistente Virtuale</span><br>
  <span>del Comune di Palazzo Adriano</span>
</div>
""", unsafe_allow_html=True)

# --- PDF & Q&A ---
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

# --- FOOTER ---
st.markdown("""
<div class="custom-footer">
    Questo assistente utilizza l’AI e potrebbe commettere errori. Verifica sempre le informazioni importanti. | © 2025 – Sviluppato da Emily D'Ugo
</div>
""", unsafe_allow_html=True)
