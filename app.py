
import streamlit as st

chiave = st.secrets["superkey"]

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI

from PIL import Image
from PyPDF2 import PdfReader

# ----- STILE GLOBALE PULITO -----
st.markdown("""
<style>
/* Barra rossa */
.top-bar {
    background-color: #d71921;
    height: 4cm;
    width: 100%;
    position: fixed;
    top: 0;
    left: 0;
    z-index: 10000;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}

/* Logo centrato sopra i titoli */
.top-bar img {
    height: 70px;
    margin-bottom: 0.5rem;
}

/* Titoli centrati */
.top-bar h1, .top-bar h2 {
    color: #ffffff;
    margin: 0;
    font-weight: 700;
    text-shadow: 1px 1px 3px rgba(0,0,0,0.2);
}

.top-bar h1 {
    font-size: 2rem;
}

.top-bar h2 {
    font-size: 1.4rem;
}

/* Spazio sotto la barra */
[data-testid="stAppViewContainer"] {
    padding-top: 5.5rem;
    background-color: #f9f9f9;
}

/* Colore testo globale */
html, body, [data-testid="stMarkdownContainer"] {
    color: #222222 !important;
    font-family: 'Helvetica Neue', sans-serif;
}

/* Input minimal */
div.stTextInput > div > input {
    background-color: white !important;
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

div.stTextInput > div > input:focus {
    border-bottom-color: #666 !important;
    outline: none !important;
}

div.stTextInput > label {
    color: #666666;
    font-weight: 500;
    font-size: 1rem;
    margin-bottom: 4px;
    display: block;
    opacity: 0.7;
    transition: opacity 0.3s ease;
}

/* Footer */
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

<!-- Barra rossa con logo e titoli -->
<div class="top-bar">
    <img src="Palazzo_Adriano-Stemma.png" />
    <h1>Assistente Virtuale</h1>
    <h2>del Comune di Palazzo Adriano</h2>
</div>
""", unsafe_allow_html=True)

# ----- PDF + LLM -----
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

# ----- FOOTER -----
st.markdown("""
<div class="custom-footer">
    Questo assistente utilizza l’AI e potrebbe commettere errori. Verifica sempre le informazioni importanti. | © 2025 – Sviluppato da Emily D'Ugo
</div>
""", unsafe_allow_html=True)
