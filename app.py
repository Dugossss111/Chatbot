
import streamlit as st

chiave = st.secrets["superkey"]

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI
#____________________________________________
import streamlit as st
from PIL import Image

st.markdown("""
    <style>
    /* Barra rossa */
    .top-bar {
        background-color: #d71921;  /* rosso vivo */
        height: 4cm;  /* altezza barra */
        width: 100%;
        position: fixed;
        top: 0;
        left: 0;
        z-index: 9999;
    }

    /* Spazio sotto la barra per contenuto */
    [data-testid="stAppViewContainer"] {
        padding-top: 14rem; /* meno spazio per avvicinare logo alla barra */
        background-color: #f9f9f9;
    }

    /* Titoli */
    h1, h2 {
        color: #222222 !important;
        text-align: center;
        margin: 0;
    }

    h1 {
        font-size: 2.6rem;
        margin-top: 10px;
        margin-bottom: 5px;
    }

    h2 {
        font-size: 2rem;
        margin-bottom: 30px;
    }

    /* Bottone */
    div.stButton > button {
        background-color: #d71921 !important;
        color: white !important;
        border-radius: 8px !important;
    }

    /* Centrare logo e spostare più a destra */
    .logo-container {
        display: flex;
        justify-content: flex-start; /* per spostare a sinistra/modificabile */
        padding-left: 3cm; /* sposta 3 cm verso destra */
        margin-bottom: 10px;
    }

    /* Input box personalizzata */
    div.stTextInput > div > input {
        background-color: #ffffff;
        border: 1.5px solid #d71921;
        border-radius: 8px;
        padding: 10px 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.15);
        font-size: 1.1rem;
        color: #222222;
    }

    /* Placeholder testo personalizzato */
    div.stTextInput > div > input::placeholder {
        color: #d71921;
        font-style: italic;
    }

    /* Label input (testo introduttivo) */
    label[for="widget"] {
        color: #222222;
        font-weight: 600;
        font-size: 1.1rem;
        margin-bottom: 5px;
        display: block;
    }
    </style>

    <div class="top-bar"></div>
""", unsafe_allow_html=True)

# Logo centrato ma spostato 3cm verso destra (grazie a container flex e padding-left)
st.markdown('<div class="logo-container">', unsafe_allow_html=True)
st.image("Palazzo_Adriano-Stemma.png", width=120)
st.markdown('</div>', unsafe_allow_html=True)

# Titolo centrato
st.markdown("<h1>Assistente Virtuale</h1>", unsafe_allow_html=True)
st.markdown("<h2>del Comune di Palazzo Adriano</h2>", unsafe_allow_html=True)

#___________________________________________
#st.image("Palazzo_Adriano-Stemma.png", width=100)


#per aggiungere il titolo
#st.header("Assistente Virtuale del Comune di Palazzo Adriano")

#from PIL import Image
#logo = Image.open("Chatbot.webp")
#st.image(logo)

#with st.sidebar:
  #per mettere il titolo alla sidebar
  #st.title("Carica i tuoi documenti")
  #per caricare il documento
  #file = st.file_uploader("Carica il tuo file", type="pdf")

file = "Costituzione_della_Repubblica_italiana.pdf"

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
    #_______________________________________________
    if domanda:
        st.write("Sto cercado le informazioni che mi hai richiesto...")
        rilevanti = vector_store.similarity_search(domanda)
        
        # Definiamo l'LLM
        llm = ChatOpenAI(
            openai_api_key = chiave,
            temperature = 1.0,
            max_tokens = 1000,
            model_name = "gpt-3.5-turbo-0125")
        # https://platform.openai.com/docs/models/compare

        # Output
        # Chain: prendi la domanda, individua i frammenti rilevanti,
        # passali all'LLM, genera la risposta

        chain = load_qa_chain(llm, chain_type="stuff")
        risposta = chain.run(input_documents = rilevanti, question = domanda)
        st.write(risposta)


