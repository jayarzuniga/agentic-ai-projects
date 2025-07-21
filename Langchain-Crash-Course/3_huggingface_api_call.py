import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv(override=True)

hf_token = os.environ["HUGGING_FACE_TOKEN"]

client = InferenceClient(
    provider="auto",
    api_key=hf_token,
)

completion = client.chat.completions.create(
    model="moonshotai/Kimi-K2-Instruct",
    messages=[
        {
            "role": "user",
            "content": "What is the capital of France?"
        }
    ],
)

print(completion.choices[0].message.content)