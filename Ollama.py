import ollama

response = ollama.chat(
    model="llama3.2",
    messages=[{"role": "user", "content": "Tell me a short fun fact about space."}],
    stream=True
)

for chunk in response:
    print(chunk["message"]["content"], end="", flush=True)

print()  # newline at the end