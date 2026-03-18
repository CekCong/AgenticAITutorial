import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec


load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "pinecone-chatbot"
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")

def ingest_docs():
    print("\nIngest Docs...\n")
    if index_name in pc.list_indexes().names():
        pc.delete_index(index_name)

    pc.create_index(
        name=index_name,
        dimension=384,        
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )


    loader = DirectoryLoader('files', glob="**/*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()

    split_docs = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
    ).split_documents(docs)

    return PineconeVectorStore.from_documents(
        documents=split_docs,
        embedding=embeddings,
        index_name=index_name
    )



def get_vectorstore():
    print("\nGet Vector From Database...\n")
    return PineconeVectorStore(index_name=index_name,embedding=embeddings)
    
    

def get_retriever(k=5):
    return get_vectorstore().as_retriever(search_kwargs={"k":k})  #k is the number of documents to retrieve

if __name__ == "__main__":
    ingest_docs()







