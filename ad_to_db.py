import os
import shutil
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import MyEmbeddingFunction
from langchain_community.vectorstores import Chroma


CHROMA_PATH = "chroma"
DATA_PATH = "data"


def main_data():
    
    print("✨ Clearing Database")
    clear_database()

    # Create (or update) the data store.
    documents = load_documents()
    chunks = split_documents(documents)
    db_chroma = Chroma.from_documents(documents=chunks, embedding=MyEmbeddingFunction, persist_directory=CHROMA_PATH)
    return db_chroma


def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()


def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)




def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


