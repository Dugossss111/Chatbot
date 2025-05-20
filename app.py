
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

/* Input box personalizzata */
div.stTextInput > div > input {
    background-color: #ffffff !important;         /* sfondo bianco */
    border: 2px solid #d71921 !important;         /* bordo rosso */
    border-radius: 8px;
    padding: 10px 15px;
    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
    font-size: 1.1rem;
    color: #444444 !important;                    /* testo più chiaro */
}


/* Label input */
div.stTextInput > label {
    color: #222222;
    font-weight: 600;
    font-size: 1.1rem;
    margin-bottom: 5px;
    display: block;
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
<h1>Assistente Virtuale</h1>
<h2>del Comune di Palazzo Adriano</h2>
""", unsafe_allow_html=True)

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
    #_______________________________________________
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
   
   
   
   
   
   
   
   
   
   
   # if domanda:
    #    st.write("Sto cercado le informazioni che mi hai richiesto...")
     #   rilevanti = vector_store.similarity_search(domanda)

        # Definiamo l'LLM
      #  llm = ChatOpenAI(
       #     openai_api_key = chiave,
        #    temperature = 1.0,
         #   max_tokens = 1000,
          #  model_name = "gpt-3.5-turbo-0125")
        # https://platform.openai.com/docs/models/compare

        # Output
        # Chain: prendi la domanda, individua i frammenti rilevanti,
        # passali all'LLM, genera la risposta

        #chain = load_qa_chain(llm, chain_type="stuff")
        #risposta = chain.run(input_documents = rilevanti, question = domanda)
        #st.write(risposta)

# Footer fisso in basso, sempre visibile, con testo semplice e senza sfondo
st.markdown("""
<style>
.footer {
    position: fixed;
    bottom: 0;
    left: 0;
    width: 100%;
    font-size: 0.8rem;
    text-align: center;
    color: #666666;
    padding: 10px;
    z-index: 9999;
}
</style>

<div class="footer">
Questo assistente utilizza l’AI e potrebbe commettere errori. Verifica sempre le informazioni importanti.<br>
© 2025 – Sviluppato da Emily D'Ugo
</div>
""", unsafe_allow_html=True)



