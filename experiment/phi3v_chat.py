import re
import json
import torch
import numpy as np
from pathlib import Path
import tvm

from mlc_llm.serve import data
from mlc_llm.serve import engine_utils
from mlc_llm.serve.sync_engine import EngineConfig, SyncMLCEngine
from mlc_llm.tokenizers import Tokenizer
from mlc_llm.protocol.mlc_chat_config import MLCChatConfig
from mlc_llm.protocol.generation_config import GenerationConfig
from mlc_llm.conversation_template import ConvTemplateRegistry

def get_test_image_2(config) -> data.ImageData:
    url = "https://www.ilankelman.org/stopsigns/australia.jpg"
    #url = "https://www.ilankelman.org/music/mela08.jpg"
    #url = "https://www.ilankelman.org/themes/birdaukcliffs.jpg"
    #url = "https://i.redd.it/pr71wuygktj11.png"
    return data.ImageData.phi3v_from_url(url, config)


if __name__ == "__main__":
    model = "HF://mengshyu/Phi-3-vision-128k-instruct-q4f16_1-MLC"

    engine = SyncMLCEngine(
        model=model,
        mode="server",
        engine_config=EngineConfig(max_total_sequence_length=4096),
    )
    max_tokens = 256

    with open("./dist/Phi-3-vision-128k-instruct-q4f16_1-MLC/mlc-chat-config.json", "r", encoding="utf-8") as file:
        model_config = json.load(file)

    prompts = [
        [data.TextData("<|user|>\nWhat is the meaning of life?<|end|>\n<|assistant|>\n")],
        [
            data.TextData("<|user|>\n"),
            get_test_image_2(model_config),
            data.TextData("\nWhat is shown in this image?<|end|>\n<|assistant|>\n"),
        ],
    ]

    output_texts, _ = engine.generate(
        prompts, GenerationConfig(max_tokens=max_tokens, stop_token_ids=[32007])
    )

    for req_id, outputs in enumerate(output_texts):
        print(f"\nPrompt {req_id}: {prompts[req_id]}")
        if len(outputs) == 1:
            print(f"Output {req_id}:{outputs[0]}\n")
        else:
            for i, output in enumerate(outputs):
                print(f"Output {req_id}({i}):{output}\n")


