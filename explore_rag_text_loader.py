import os
from dotenv import load_dotenv
from datasets import load_dataset
from langchain_openai import OpenAIEmbeddings
from langchain_astradb import AstraDBVectorStore

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

# load and process file
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
src_file_name = os.path.join(BASE_DIR, "docs", "foods.csv")

lines = [
    line.strip()
    for line in open(src_file_name).readlines()
    if line.split(",")[0] != "id"
]

ids = []
foods = []
for line in lines:
    id, food = line.split(",")
    ids.append(id)
    foods.append(food)
    
vstore.add_texts(texts=foods, ids=ids)