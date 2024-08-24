from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser 
from langchain.schema.runnable import RunnableLambda, RunnableParallel

# Initialize the model
model = ChatOpenAI()

# Define the prompt template
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert product reviewer."),
        ("human", "List the main fetures of the product {product}.")
    ]
)

# Define the pros analysis step
def analyze_pros(features):
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert product reviewer."),
            ("human", "Given these features: {features}, list the pros.")
        ]
    )
    return prompt_template.format_prompt(features=features)

# Define the cons analysis step
def analyze_cons(features):
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert product reviewer."),
            ("human", "Given these features: {features}, list the cons.")
        ]
    )
    return prompt_template.format_prompt(features=features)

def combine_pros_cons(pros, cons):
    return f"Pros:\n{pros}\n\nCons:\n{cons}"

# Define the pros chain
pros_chain = (
    RunnableLambda(lambda x: analyze_pros(x)) | model | StrOutputParser()
)

# Define the cons chain
cons_chain = (
    RunnableLambda(lambda x: analyze_cons(x)) | model | StrOutputParser()
)

# Define the main chain
chain = (
    prompt_template
    | model
    | StrOutputParser()
    | RunnableParallel(branches={"pros": pros_chain, "cons": cons_chain})
    | RunnableLambda(lambda x: combine_pros_cons(x["branches"]["pros"], x["branches"]["cons"]))
)

result = chain.invoke({"product": "iPhone 15"})
print(result)