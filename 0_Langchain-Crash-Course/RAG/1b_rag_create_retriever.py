import os

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

db = Chroma(
    embeddings=embeddings,
    persist_directory=persistent_directory,
)

query = "Who is Odysseus wife?"

retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.4},
)

relevant_docs = retriever.invoke(query)

print("\n----Relevant Documents---")
for i, doc in enumerate(relevant_docs):
    print(f"Document {i}: {doc.page_content}")
    if doc.metadata:
        print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")