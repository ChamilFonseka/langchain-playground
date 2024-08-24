from langchain.schema.runnable import RunnableLambda, RunnableParallel, RunnablePassthrough

def add_one(x):
    return x + 1

def multiply_by_two(x):
    return x * 2

chain = RunnableParallel(
    add_one=RunnableLambda(lambda x: add_one(x['num'])),
    multiply_by_two=RunnableLambda(lambda x: multiply_by_two(x['num'])),
    extra=RunnablePassthrough.assign(multiply_by_three=lambda x: x["num"] * 3) # This passes the input data through unchanged but adds a new key-value pair
)
 
result = chain.invoke({'num': 2})
print(result)
