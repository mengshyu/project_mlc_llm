from mlc_llm import MLCEngine

# Create engine
#model = "HF://mlc-ai/Phi-3-mini-128k-instruct-q4f16_1-MLC"
#model = "HF://mengshyu/llava-1.5-7b-hf-q4f16_1-MLC"
model = "HF://mengshyu/Phi-3-vision-128k-instruct-q0f16-MLC"
#model = "HF://mengshyu/Phi-3-mini-128k-instruct-q4f16_1-MLC"
url = "https://www.ilankelman.org/stopsigns/australia.jpg"
engine = MLCEngine(model)

# Run chat completion in OpenAI API.
for response in engine.chat.completions.create(
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": url ,
            },
            {"type": "text", "text": "What does this image represent?"},
        ],
    }],
    model=model,
    stream=True,
):
    for choice in response.choices:
        print(choice.delta.content, end="", flush=True)
print("\n")

engine.terminate()
