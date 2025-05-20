
import streamlit as st

chiave = st.secrets["superkey"]

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI

import streamlit as st
from PIL import Image

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
}

/* Spazio sotto la barra */
[data-testid="stAppViewContainer"] {
    padding-top: 4.5rem;
    background-color: #f9f9f9;
}

/* Titoli */
h1, h2 {
    color: #222222 !important;
    text-align: center;
    margin: 0;                  /* elimina margini */
    font-weight: 700;
    text-shadow: 1px 1px 2px rgba(255,255,255,0.6);
}

h1 {
    font-size: 2.6rem;
    margin-bottom: 0 !important;  /* nessun margine sotto h1 */
}

h2 {
    font-size: 2rem;
    margin-top: 0 !important;     /* nessun margine sopra h2 */
    margin-bottom: 0 !important;
}


}
/* Colore del testo dell'output */
html, body, [data-testid="stMarkdownContainer"] {
    color: #222222 !important;
}

</style>

<div class="top-bar"></div>
""", unsafe_allow_html=True)

# Usare colonne per spostare il logo più a destra
col1, col2, col3 = st.columns([2.4, 1.2, 2.4])
with col2:
    st.image("Palazzo_Adriano-Stemma.png", width=120)

st.markdown("""
<h1 style="position: relative; z-index: 10000;">Assistente Virtuale</h1>
<h2 style="position: relative; z-index: 10000;">del Comune di Palazzo Adriano</h2>
""", unsafe_allow_html=True)

file = "Comune_di_Palazzo_Adriano.pdf"

from PyPDF2 import PdfReader

if file is not None:
    testo_letto = PdfReader(file)

    testo = ""
    for pagina in testo_letto.pages:
        testo = testo + pagina.extract_text()
        # st.write(testo)

    # Usiamo il text splitter di Langchain
    testo_spezzato = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=1000, # Numero di caratteri per chunk
        chunk_overlap=150,
        length_function=len
        )

    pezzi = testo_spezzato.split_text(testo)
    #st.write(pezzi)

    # Generazione embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=chiave)

    # Vector store - FAISS (by Facebook)
    vector_store = FAISS.from_texts(pezzi, embeddings)

    # Prompt
    domanda = st.text_input("Digita qui la tua richiesta:")

if domanda:
    with st.spinner("Sto cercando le informazioni che mi hai richiesto..."):
        rilevanti = vector_store.similarity_search(domanda)

        # Definiamo l'LLM
        llm = ChatOpenAI(
            openai_api_key=chiave,
            temperature=1.0,
            max_tokens=1000,
            model_name="gpt-3.5-turbo-0125"
        )

        chain = load_qa_chain(llm, chain_type="stuff")
        risposta = chain.run(input_documents=rilevanti, question=domanda)

    st.write(risposta)

/* Input box minimal, neutro come questa chat */

div.stTextInput > div > input {
    background-color: transparent !important;
    border: none !important;
    border-bottom: 1px solid #ccc !important;
    border-radius: 0 !important;
    padding: 8px 0 6px 0;
    font-size: 1.1rem;
    color: #333 !important;
    box-shadow: none !important;
    outline: none !important;
    transition: border-color 0.3s ease;
}
st.markdown("""
<style>
/* Focus: linea sottile diventa più scura */
div.stTextInput > div > input:focus {
    border-bottom-color: #666 !important;
    outline: none !important;
}

/* Label input minimal */
div.stTextInput > label {
    color: #666666;
    font-weight: 500;
    font-size: 1rem;
    margin-bottom: 4px;
    display: block;
    opacity: 0.7;
    transition: opacity 0.3s ease;
}


# Footer fisso in basso, sempre visibile, con testo semplice e senza sfondo
st.markdown("""
<style>
/* Footer fisso in basso senza sfondo */
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

<div class="custom-footer">
    Questo assistente utilizza l’AI e potrebbe commettere errori. Verifica sempre le informazioni importanti. | © 2025 – Sviluppato da Emily D'Ugo
</div>
""", unsafe_allow_html=True)

