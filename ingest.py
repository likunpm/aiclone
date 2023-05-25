"""Load html from files, clean up, split, ingest into Weaviate."""
import pickle

from langchain.document_loaders import ReadTheDocsLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
#2023-05-12 17:47:02 datali修改 
from langchain.document_loaders import CSVLoader


def ingest_docs():
    """Get documents from web pages."""
 #  loader = ReadTheDocsLoader("langchain.readthedocs.io/en/latest/", errors="ignore") 
    
    loader = CSVLoader("doc_of_me/me.csv")
    print("----loader:",loader)
    raw_documents = loader.load()
    print("----raw_documents:",raw_documents)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    documents = text_splitter.split_documents(raw_documents)
    print("----documents:",documents)
    embeddings = OpenAIEmbeddings()
    print("----embeddings:",embeddings)
    vectorstore = FAISS.from_documents(documents, embeddings)

    # Save vectorstore
    with open("vectorstore.pkl", "wb") as f:
        pickle.dump(vectorstore, f)


if __name__ == "__main__":
    ingest_docs()
