import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_astradb import AstraDBVectorStore
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser 
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

load_dotenv()

ASTRA_DB_APPLICATION_TOKEN = os.environ.get("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_API_ENDPOINT = os.environ.get("ASTRA_DB_API_ENDPOINT")
OPEN_AI_API_KEY = os.environ.get("OPENAI_API_KEY")
ASTRA_DB_COLLECTION = "gen_ai_john_doe"

current_dir = os.path.dirname(os.path.realpath(__file__))
file_path = os.path.join(current_dir, "docs", "biography-of-john-doe.txt")

if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")

# Load the document
loader = TextLoader(file_path)
documents = loader.load()

# Add metadata to the document, astradb will add the source metadata by default.
# No need to add it manually.
# for document in documents:
#     document.metadata = {"source": file_path}


# Split the document into chunks
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
document_chunks   = text_splitter.split_documents(documents)

# Initialize the embedding model
embedding = OpenAIEmbeddings()

# Initialize the vector store
vstore = AstraDBVectorStore(
    embedding=embedding,
    collection_name=ASTRA_DB_COLLECTION,
    token=ASTRA_DB_APPLICATION_TOKEN,
    api_endpoint=ASTRA_DB_API_ENDPOINT,
)

# Check if the collection is empty, then add the documents
if vstore.collection.count_documents()['status']['count'] == 0:
    vstore.add_documents(document_chunks)

similarity_search_query = "Who is Emily Rivers?"

# Perform similarity search using the vector store
relevant_docs = vstore.similarity_search(similarity_search_query, k=2)  
for (doc) in relevant_docs:
    print(doc.page_content)
print("-----------------------------------------------------------")

# Initialize the retriever
retriever = vstore.as_retriever(
    serch_type="similarity_score_threshold",
    search_kwargs = {"k": 2, "score_threshold": 0.8}
)

# Perform retrieval using the retriever
relevant_docs = retriever.invoke(similarity_search_query)
for doc in relevant_docs:
    print(doc.page_content)
print("-----------------------------------------------------------")   

# Define the prompt template
prompt_template = """
You are a friend of John Doe.
Context: {context}.
Question: {question}?"""

prompt = ChatPromptTemplate.from_template(prompt_template)

model = ChatOpenAI()

# Define the chain
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model 
    | StrOutputParser()
)

print(chain.invoke("Who are the members of John's family?"))
print(chain.invoke("What was John's wife's job?"))
print(chain.invoke("What was the John's first book?"))