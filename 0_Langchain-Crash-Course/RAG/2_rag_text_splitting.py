import os

from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
    TextSplitter,
    TokenTextSplitter,
)
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "books", "romeo_and_juliet.txt")
db_dir = os.path.join(current_dir, "db")

if not os.path.exists(file_path):
    raise FileNotFoundError(
        f"The file {file_path} does not exist. Please check the path."
    )

loader = TextLoader(file_path)
documents = loader.load()

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)

def create_vector_store(docs, store_name):

    persistent_directory = os.path.join(db_dir, store_name)
    if not os.path.exists(persistent_directory):
        print(f"\nCreating vector store {store_name}...")
        db = Chroma.from_documents(
            docs, embeddings, persist_directory=persistent_directory
        )
        print(f"Vector store {store_name} created.")
    else:
        print(f"\nVector store {store_name} already exists.")


print ("\n---- Using Character-based Splitting ----")
char_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
char_docs = char_splitter.split_documents(documents)
create_vector_store(char_docs, "chroma_db_char")


print ("\n --- Using Sentence-based Splitting ---")
sent_splitter = SentenceTransformersTokenTextSplitter(chunk_size=1000)
sent_docs = sent_splitter.split_documents(documents)
create_vector_store(sent_docs, "chroma_db_sent")


print ("\n --- Using Token-based Splitting ---")
token_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=0)
token_docs = token_splitter.split_documents(documents)
create_vector_store(token_docs, "chroma_db_token")


print ("\n --- Using Recursive Splitting ---")
recursive_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
recursive_docs = recursive_splitter.split_documents(documents)
create_vector_store(recursive_docs, "chroma_db_recursive")


print ("\n --- Using Custom Text Splitter ---")
class CustomTextSplitter(TextSplitter):
    def split_text (self, text):
        return text.split("\n\n")

custom_splitter = CustomTextSplitter()
custom_docs = custom_splitter.split_documents(documents)
create_vector_store(custom_docs, "chroma_db_custom")


def query_vector_store(store_name, query):
    persistent_directory = os.path.join(db_dir, store_name)
    if os.path.exists(persistent_directory):
        print(f"\nUsing vector store {store_name}...")
        db = Chroma(
            embedding_function=embeddings,
            persist_directory=persistent_directory,
        )
        retriever = db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 3, "score_threshold": 0.1},
        )
        relevant_docs = retriever.invoke(query)
        print("\n----Relevant Documents---")
        for i, doc in enumerate(relevant_docs):
            print(f"Document {i}: {doc.page_content}")
            if doc.metadata:
                print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
    else:
        print(f"\nVector store {store_name} does not exist.")

query = "How Juliet Dies?"

query_vector_store("chroma_db_char", query)
query_vector_store("chroma_db_sent", query)
query_vector_store("chroma_db_token", query)
query_vector_store("chroma_db_recursive", query)
query_vector_store("chroma_db_custom", query)
