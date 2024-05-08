import openai
import os

# optional; defaults to `os.environ['OPENAI_API_KEY']`
openai.api_key = os.environ['OPENAI_API_KEY']

completion = openai.chat.completions.create(
    model="gpt-4-1106-preview",
    messages=[
        {
            "role": "user",
            "content": "How do I output all files in a directory using Python?",
        },
    ],
)
print(completion.choices[0].message.content)