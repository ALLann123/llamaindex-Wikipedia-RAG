#!/usr/bin/python3
import streamlit as st
from dotenv import load_dotenv
import os 

from llama_index.llms.groq import Groq
from llama_index.readers.wikipedia import WikipediaReader
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Load environment variables
load_dotenv()
api_key = os.getenv("GITHUB_TOKEN")

# Create the LangChain chat model using the GitHub Marketplace endpoint
# Initialize the Groq LLM
llm = Groq(
    model="llama3-70b-8192",  # Model name must be specified
    api_key=os.getenv("GROQ_API_KEY"),  # Use 'api_key' instead of 'groq_api_key'
    temperature=0
)

#our rag pdf file
INDEX_DIR='wiki_rag'
#Pages on our RAG ie will retrieve information from wikipedia
PAGES = [
    "Artificial intelligence",
    "Machine learning",
    "Deep learning",
    "Neural network",
    "Convolutional neural network",
    "Natural language processing",  # more accurate than "Natural Language"
    "Generative artificial intelligence",  # instead of "Gen AI"
    "Large language model",
    "Automation",  # corrected
    "Intelligent agent",  # instead of "Introduction to AI agents"
    "Multi-agent system",  # instead of "Agent frameworks"
    "Applications of artificial intelligence",
    "AI in finance"
]


#creating the catche
@st.cache_resource
def get_index():

    #check if we have our RAG file and load it
    if os.path.isdir(INDEX_DIR):
        storage=StorageContext.from_defaults(persist_dir=INDEX_DIR)
        return load_index_from_storage(storage)
    
    #create it if it does nt exist
    #now if the file is available we load our document from wikipedia
    docs=WikipediaReader().load_data(pages=PAGES, auto_suggest=False)
    #create the embedding model
    embeddings =HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    #create a vector store index
    index=VectorStoreIndex.from_documents(docs, embed_model=embeddings)
    #lets persist our text embeddings to disk
    index.storage_context.persist(persist_dir=INDEX_DIR)

    return index

@st.cache_resource
def get_query_engine():
    index=get_index()
    
    #now the retriever using our llm to retrieve the top 3 results
    return index.as_query_engine(llm=llm, similarity_top_k=3)

#---Build the user interface

def main():
    st.set_page_config(
        page_title="Wikipedia RAG App",  # This sets the tab title
        page_icon="📘",                 # Optional: sets the favicon
        layout="centered"               # Optional: layout setting
    )
    st.title("🤖Wikipedia RAG Application")

    question=st.text_input("Ask a question")

    if st.button('Submit') and question:
        with st.spinner('Thinking...'):
            qa=get_query_engine()
            response=qa.query(question)

        st.subheader('Answer')
        st.write(response.response)

        st.subheader('Retrieved context')

        for src in response.source_nodes:
            st.markdown(src.node.get_content())


if __name__=='__main__':
    main()

