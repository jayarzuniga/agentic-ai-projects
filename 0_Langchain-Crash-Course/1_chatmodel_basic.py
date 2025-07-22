# from dotenv import load_dotenv
# from langchain_openai import ChatOpenAI

# load_dotenv()

# model = ChatOpenAI(
#     model_name="gpt-4o-mini"
# )

# result = model.invoke("What is 81 divided by 9?")
# print (f"Full result: {result}")
# print (f"Content only: {result.content}")

from langchain_community.chat_models import ChatOllama

model = ChatOllama(model="deepseek-r1:latest")

result = model.invoke("What is 81 divided by 9?")

print(f"Full result: {result}")
print(f"Content only: {result.content}")
