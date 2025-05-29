
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
    border: 2px solid #d71921 !important; /* bordo rosso fisso */
    border-radius: 6px !important;
    padding: 10px !important;
    font-size: 16px !important;
    outline: none !important;
    box-shadow: none !important;
    caret-color: #d71921 !important;
}

/* Mantieni bordo rosso anche quando è in focus o no */
input[type="text"]:focus,
input[type="text"]:not(:focus) {
    border: 2px solid #d71921 !important;
    box-shadow: none !important;
    outline: none !important;
}

/* Placeholder visibile e ben leggibile */
input::placeholder {
    color: #444444 !important;
    opacity: 1 !important;
    font-weight: 600 !important;
    text-shadow: none !important;

}

</style>

<div class="top-bar">
    <img src="Palazzo_Adriano-Stemma.png" alt="Logo Comune">
</div>
""", unsafe_allow_html=True)

# Layout colonne e logo centrato
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

# Lista FAQ
faq = [
    "Quali sono gli orari di apertura del Comune?",
    "Come posso richiedere un certificato anagrafico?",
    "Dove si trova l'ufficio tributi?",
    "Come prenotare un appuntamento con l'ufficio tecnico?"
]

if "faq_question" not in st.session_state:
    st.session_state.faq_question = ""

# Mostra pulsanti FAQ in orizzontale
cols = st.columns(len(faq))
for i, q in enumerate(faq):
    if cols[i].button(q):
        st.session_state.faq_question = q

# Input con domanda precompilata (da FAQ)
domanda = st.text_input(
    "Ciao, sono Pai. In cosa posso esserti utile?",
    value=st.session_state.get("faq_question", "")
)

# Se l'utente modifica manualmente la domanda, resetta FAQ precompilata
if domanda != st.session_state.get("faq_question", ""):
    st.session_state.faq_question = ""

st.markdown('</div>', unsafe_allow_html=True)

# Cronologia chat in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

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

        # Aggiorna cronologia chat
        st.session_state.chat_history.append((domanda, risposta))

        st.write(risposta)

# Mostra cronologia chat
st.markdown("---")
st.markdown("## Conversazione")
for q, a in st.session_state.chat_history:
    st.markdown(f"**Tu:** {q}")
    st.markdown(f"**Pai:** {a}")
    st.markdown("---")

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

