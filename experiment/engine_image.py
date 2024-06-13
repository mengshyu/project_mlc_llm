from mlc_llm import MLCEngine

import json
from pathlib import Path

from mlc_llm.protocol.generation_config import GenerationConfig
from mlc_llm.serve import data
from mlc_llm.serve.sync_engine import EngineConfig, SyncMLCEngine

model = "HF://mengshyu/llava-1.5-7b-hf-q4f16_1-MLC"

def get_test_image(config) -> data.ImageData:
    #url = "https://llava-vl.github.io/static/images/view.jpg"
    #url = "https://heronscrossing.vet/wp-content/uploads/Golden-Retriever-1536x1024.jpg"
    url = "https://www.littleflowergoldens.com/wp-content/uploads/2017/07/benny-100x667.jpg"
    return data.ImageData.from_url(url, config)


def test_engine_generate(model):
    # Create engine
    engine = SyncMLCEngine(
        model=model,
        mode="server",
        engine_config=EngineConfig(max_total_sequence_length=4096),
    )
    max_tokens = 256

    with open("./dist/llava-1.5-7b-hf-q4f16_1-MLC/mlc-chat-config.json", "r", encoding="utf-8") as file:
        model_config = json.load(file)

    prompts = [
        [
            data.TextData("USER: "),
            get_test_image(model_config),
            data.TextData("\nWhat does this image represent? ASSISTANT:"),
        ],
        [
            data.TextData("USER: "),
            get_test_image(model_config),
            data.TextData("\nIs there a dog in this image? ASSISTANT:"),
        ],
        [data.TextData("USER: What is the meaning of life? ASSISTANT:")],
    ]

    output_texts, _ = engine.generate(
        prompts, GenerationConfig(max_tokens=max_tokens, stop_token_ids=[2])
    )

    for req_id, outputs in enumerate(output_texts):
        print(f"\nPrompt {req_id}: {prompts[req_id]}")
        if len(outputs) == 1:
            print(f"Output {req_id}:{outputs[0]}\n")
        else:
            for i, output in enumerate(outputs):
                print(f"Output {req_id}({i}):{output}\n")


if __name__ == "__main__":
    test_engine_generate(model)

