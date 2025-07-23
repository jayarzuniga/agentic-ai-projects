import os

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv
load_dotenv(override=True)

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "books", "odyssey.txt")
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

if not os.path.exists(persistent_directory):
    print ("Persistent directory does not exist. Initializing vector store...")

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"The file {file_path} does not exist. Please check the path."
        )
    
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    print("\n----Document Chunks Information---")
    print(f"Number of document chunks: {len(docs)}")
    print(f"Sample chunk: \n{docs[0].page_content}\n")


    print("\n----Creating Vector Store---")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )
    print("\n----Embeddings created---")

    print("\n----Persisting Vector Store---")
    db = Chroma.from_documents(
        docs, embeddings, persist_directory=persistent_directory
        )
    print("\n----Vector Store persisted---")

else:
    print("Persistent directory exists. Loading vector store...")
