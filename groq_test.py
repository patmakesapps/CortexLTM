import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
model = os.getenv("GROQ_MODEL")

if not api_key:
    raise RuntimeError(
        "Could not load GROQ_API_KEY from .env...check to make sure your api key is loaded"
    )
if not model:
    raise RuntimeError("Could not load Groq model...check your .env")

client = Groq(api_key=api_key)

resp = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Say 'Groq is connected' and nothing else."}],
    temperature=0,
)

print(resp.choices[0].message.content)
