
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
    .top-bar {
        background-color: #d71921;
        height: 160px;
        width: 100%;
        position: fixed;
        top: 0;
        left: 0;
        z-index: 9999;
    }

    [data-testid="stAppViewContainer"] {
        padding-top: 180px;
        background-color: #f9f9f9;
    }

    h1, h2 {
        color: #222222 !important;
        text-align: center;
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

    div.stButton > button {
        background-color: #d71921 !important;
        color: white !important;
        border-radius: 8px !important;
    }
    </style>

    <div class="top-bar"></div>
""", unsafe_allow_html=True)

#Logo spostato più verso il centro usando colonne asimmetriche
col1, col2, col3 = st.columns([0.8, 2.2, 1])  # aggiustato per spostarlo più verso il centro
with col2:
    st.image("Palazzo_Adriano-Stemma.png", width=120)

#Titolo centrato sotto il logo
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


