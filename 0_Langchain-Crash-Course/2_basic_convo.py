# from dotenv import load_dotenv
# from langchain_openai import ChatOpenAI

# load_dotenv()

# model = ChatOpenAI(
#     model_name="gpt-4o-mini"
# )

# result = model.invoke("What is 81 divided by 9?")
# print (f"Full result: {result}")
# print (f"Content only: {result.content}")

# from langchain_community.chat_models import ChatOllama

# model = ChatOllama(model="deepseek-r1:latest")

# messages = [
#     SystemMessage(
#         content="You are a helpful assistant that answers questions about math."
#     ),
#     Humanmessage(
#         content="What is 81 divided by 9?"
#     )
# ]

# result = model.invoke(messages)

# print(f"Full result: {result}")
# print(f"Content only: {result.content}")


from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv(override=True)

deepseek_api_key = os.environ["DEEPSEEK_API_KEY"]

client = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
    ],
    stream=False
)

print(response.choices[0].message.content)