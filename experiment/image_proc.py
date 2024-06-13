import json
from pathlib import Path

from mlc_llm.protocol.generation_config import GenerationConfig
from mlc_llm.serve import data
from mlc_llm.serve.sync_engine import EngineConfig, SyncMLCEngine


def get_test_image(config) -> data.ImageData:
    url = "https://llava-vl.github.io/static/images/view.jpg"
    return data.ImageData.gen_phi3_image_embed_from_url(url, config)
    #return data.ImageData.from_url(, config)


model = "dist/llava-1.5-7b-hf-q4f16_1-MLC/"
with open(Path(model) / "mlc-chat-config.json", "r", encoding="utf-8") as file:
    print("get image")
    model_config = json.load(file)
    get_test_image(model_config)



