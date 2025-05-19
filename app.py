
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

# CSS personalizzato
st.markdown("""
    <style>
    /* Sfondo chiaro */
    [data-testid="stAppViewContainer"] {
        background-color: #f9f9f9;
    }
    /* Centrare immagine logo */
    .logo-container {
        display: flex;
        justify-content: center;
        margin-bottom: 10px;
    }
    /* Titolo principale */
    .main-title {
        color: #222222;
        text-align: center;
        font-weight: 700;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 2.8rem;
        margin-bottom: 0;
    }
    /* Sottotitolo rosso con a capo */
    .subtitle-red {
        color: #c00000;  /* rosso comune */
        text-align: center;
        font-weight: 600;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 1.8rem;
        margin-top: 0;
        margin-bottom: 30px;
    }
    /* Bottone personalizzato */
    div.stButton > button {
        background-color: #c00000;
        color: white;
        border-radius: 8px;
        padding: 0.5em 1em;
        font-size: 1em;
        border: none;
        cursor: pointer;
    }
    </style>
""", unsafe_allow_html=True)

# Logo centrato
logo = Image.open("Palazzo_Adriano-Stemma.png")
st.markdown('<div class="logo-container">', unsafe_allow_html=True)
st.image(logo, width=100)
st.markdown('</div>', unsafe_allow_html=True)

# Titolo su due righe
st.markdown('<h1 class="main-title">Assistente Virtuale del</h1>', unsafe_allow_html=True)
st.markdown('<h2 class="subtitle-red">del Comune di Palazzo Adriano</h2>', unsafe_allow_html=True)

#___________________________________________
st.image("Palazzo_Adriano-Stemma.png", width=100)


#per aggiungere il titolo
st.header("Assistente Virtuale del Comune di Palazzo Adriano")

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
    domanda = st.text_input("Chiedi al chatbot:")
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


