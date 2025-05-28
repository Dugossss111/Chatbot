
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
/* Contenitore del logo, posizionato sopra tutto */
.logo-top {
    position: fixed;
    top: 0;
    left: 0;
    height: 4cm;   /* altezza logo */
    width: auto;
    z-index: 10001;  /* sopra la barra */
    padding: 0.3rem 1rem;
}

/* Barra rossa fissa sotto il logo */
.top-bar {
    background-color: #d71921;
    height: 3.5cm;
    width: 100%;
    position: fixed;
    top: 4cm;  /* parte subito sotto il logo */
    left: 0;
    z-index: 9999;
}

/* Spazio sotto barra per contenuto */
[data-testid="stAppViewContainer"] {
    padding-top: 8.5rem;  /* spazio per logo + barra */
    background-color: #f9f9f9;
}
</style>

<div class="logo-top">
    <img src="Palazzo_Adriano-Stemma.png" alt="Logo Comune" style="height: 100%;">
</div>

<div class="top-bar"></div>
""", unsafe_allow_html=True)

# Layout colonne e logo centrato (immagine grande centrale sotto la barra)
col1, col2, col3 = st.columns([2.4, 1.2, 2.4])
with col2:
    st.image("noname.png", width=600)

# Titoli sopra la barra (ma sotto la barra rossa fissa)
st.markdown("""
<h1 style="position: relative; z-index: 10000;">Assistente Virtuale</h1>
<h2 style="position: relative; z-index: 10000;">del Comune di Palazzo Adriano</h2>
""", unsafe_allow_html=True)

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
<div class="custom-footer">
    Questo assistente utilizza l’AI e potrebbe commettere errori. Verifica sempre le informazioni importanti. | © 2025 – Sviluppato da Emily D'Ugo
</div>
""", unsafe_allow_html=True)
