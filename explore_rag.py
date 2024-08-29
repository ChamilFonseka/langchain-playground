import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_astradb import AstraDBVectorStore
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser 
from langchain.schema.runnable import RunnablePassthrough

load_dotenv()

ASTRA_DB_APPLICATION_TOKEN = os.environ.get("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_API_ENDPOINT = os.environ.get("ASTRA_DB_API_ENDPOINT")
OPEN_AI_API_KEY = os.environ.get("OPENAI_API_KEY")
ASTRA_DB_COLLECTION = "gen_ai_family"

embedding = OpenAIEmbeddings()

vstore = AstraDBVectorStore(
    embedding=embedding,
    collection_name=ASTRA_DB_COLLECTION,
    token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
    api_endpoint=os.environ["ASTRA_DB_API_ENDPOINT"],
)

if vstore.collection.count_documents()['status']['count'] == 0:

    chamils_description = """
    Chamil Harshana Fonseka was born on September 17, 1989. He is a married man and a Software Engineer. 
    His wife's name is M.M. Shamika. They have a son and a daughter. Their son's name is Ayan Aloka Fonseka, 
    and their daughter is Ayanna Akeshi Fonseka. Chamil lives in Colombo, Sri Lanka, and drives a red 2001 Honda Civic.
    """

    shammikas_description = """
    M.M. Shamika was born on September 1, 1989. She is married to Chamil. She was an Account Executive.
    After the birth of their son, she stopped working and became a housewife.
    """

    ayans_description = """
    Ayan Aloka Fonseka was born on December 29, 2019. He is a playful boy who still goes to nursery. 
    His favorite cartoon is Sonic. He likes racing cars too and loves his younger sister Ayanna very much.
    """

    ayannas_description = """
    Ayanna Akeshi Fonseka was born on April 14, 2022. She is a cute little girl who always plays with her brother Ayan. 
    Her favorite food is chocolate.
    """

    vstore.add_texts(
        texts=[chamils_description, shammikas_description, ayans_description, ayannas_description], 
        ids=["Chamil", "Shammika", "Ayan", "Ayanna"]
    )

retriever = vstore.as_retriever()

prompt_template = """
You are a friend of Chamil.
Context: {context}.
Question: {question}?"""

prompt = ChatPromptTemplate.from_template(prompt_template)

model = ChatOpenAI()

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model 
    | StrOutputParser()
)

print(chain.invoke("Who are the members of Chamil's family?"))
print(chain.invoke("What was Chamil's wife's job?"))
print(chain.invoke("What is Ayan's favorite cartoon?"))
print(chain.invoke("What is Ayanna's favorite food?"))
print(chain.invoke("Where does Chamil live?"))
print(chain.invoke("What car does Chamil drive?"))

