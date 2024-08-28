import os
from typing import List
from dotenv import load_dotenv
from datasets import load_dataset
from langchain_openai import OpenAIEmbeddings
from langchain_astradb import AstraDBVectorStore
from langchain_core.documents import Document
from langchain.text_splitter import CharacterTextSplitter

# Load the environment variables
load_dotenv()

ASTRA_DB_APPLICATION_TOKEN = os.environ.get("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_API_ENDPOINT = os.environ.get("ASTRA_DB_API_ENDPOINT")
OPEN_AI_API_KEY = os.environ.get("OPENAI_API_KEY")
ASTRA_DB_COLLECTION = "gen_ai_food"

embedding = OpenAIEmbeddings()

vstore = AstraDBVectorStore(
    embedding=embedding,
    collection_name=ASTRA_DB_COLLECTION,
    token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
    api_endpoint=os.environ["ASTRA_DB_API_ENDPOINT"],
)

retriever = vstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.8},
)

docs = retriever.invoke("List all vegetables.")

for doc in docs:
    print(doc.page_content)


