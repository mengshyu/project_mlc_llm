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

model_path = "./dist/Phi-3-vision-128k-instruct-q4f16_1-MLC"

def get_test_image(config) -> data.ImageData:
    #url = "https://llava-vl.github.io/static/images/view.jpg"
    url = "https://www.ilankelman.org/stopsigns/australia.jpg"
    out = data.ImageData.gen_phi3_image_embed_from_url(url, config)
    torch.save(torch.from_numpy(out["pixel_values"]), "./dump/out_pixel_values.pt")
    return out

def convert_prompt(prompt):
    config_file_path = model_path + "/mlc-chat-config.json"
    with open(config_file_path, mode="rt", encoding="utf-8") as file:
        chat_config = MLCChatConfig.model_validate_json(file.read())
    conv_template = chat_config.conv_template

    conversation = (
            ConvTemplateRegistry.get_conv_template(conv_template)
            if isinstance(conv_template, str)
            else conv_template
    )

    with open(config_file_path, "r", encoding="utf-8") as file:
        config = json.load(file)

    tokenizer = Tokenizer(str(model_path))

    tokens = []
    conversation.messages.append(("user", None))
    parsed_prompt = conversation.as_prompt(config)
    tokens.append(engine_utils.process_prompts(parsed_prompt, tokenizer.encode)[0])

    conversation.messages.clear()
    conversation.messages.append(("assistant", prompt))
    parsed_prompt = conversation.as_prompt(config)
    tokens.append(engine_utils.process_prompts(parsed_prompt, tokenizer.encode)[0])  # type: ignore
    return tokens


def image_text_to_input(images, prompt):

    prompt_chunks = convert_prompt(prompt)
    #print("prompt:", prompt_chunks)
    if 'num_img_tokens' in images:
        num_img_tokens = images['num_img_tokens']
    else:
        assert 'num_crops' in images, 'num_crops must be provided in images if num_img_tokens is not provided'
        num_crops = images['num_crops']
        num_img_tokens = [_num_crops * self.num_img_tokens for _num_crops in num_crops]
    images, image_sizes = images['pixel_values'].astype(np.float32), np.array(images['image_sizes'], dtype=np.int32)

    #debug flow
    #images = torch.load("/ssd1/mengshiy/workspace/tmp/test_phi3vision_tir/phi3vision_dump/img_feature_proj.pt").cpu().to(torch.float32).numpy().squeeze()
    #images = np.squeeze(images)
    #print("shape of pixel valeus", images.shape)
    # image_tags needs to start from 1 to n
    user_prompt = '<|user|>\n'
    assistant_prompt = '<|assistant|>\n'
    prompt_suffix = "<|end|>\n"

    texts = f"{user_prompt}<|image_1|>\nWhat is shown in this image?{prompt_suffix}{assistant_prompt}"
    pattern = r"<\|image_\d+\|>"

    image_tags = re.findall(pattern, texts)
    image_ids = [int(s.split("|")[1].split("_")[-1]) for s in image_tags]
    unique_image_ids = sorted(list(set(image_ids)))
    assert unique_image_ids == list(range(1, len(unique_image_ids)+1)), f"image_ids must start from 1, and must be continuo    us int, e.g. [1, 2, 3], cannot be {unique_image_ids}"
    ## total images must be the same as the number of image tags
    #assert len(unique_image_ids) == len(images), f"total images must be the same as the number of image tags, got {len(unique_image_ids)} image tags and {len(images)} images"
    image_ids_pad = [[-iid]*num_img_tokens[iid-1] for iid in image_ids]

    def insert_separator(X, sep_list):
        if len(X) > len(sep_list):
            sep_list.append([])
        return [ele for sublist in zip(X, sep_list) for ele in sublist]
    input_ids = []
    offset = 0
    for x in insert_separator(prompt_chunks, image_ids_pad):
        input_ids.extend(x[offset:])
    input_ids = np.array(input_ids, dtype=np.int32).reshape(1, -1)
    attention_mask = (input_ids > -1000000).astype(np.int32)
    #print("shape of input ids", input_ids[0].shape)
    #print("shape of image sizes", image_sizes.shape)
    return {"input_ids": input_ids[0],
            "attention_mask": attention_mask,
            "pixel_values": images,
            "image_sizes": image_sizes}

def GetPhi3ImageData():
    with open(Path(model_path) / "mlc-chat-config.json", "r", encoding="utf-8") as file:
        #print("get image")
        model_config = json.load(file)
        images = get_test_image(model_config)

        prompt = "<|user|>\n"

        inputs = image_text_to_input(images, prompt)
        input_ids = tvm.nd.array(inputs["input_ids"])
        pixel_values = tvm.nd.array(inputs["pixel_values"])
        image_sizes = tvm.nd.array(inputs["image_sizes"])

        return data.ImageData(pixel_values, 1024)

if __name__ == "__main__":
    model = "HF://mengshyu/Phi-3-vision-128k-instruct-q4f16_1-MLC"

    image_data = GetPhi3ImageData()

    engine = SyncMLCEngine(
        model=model,
        mode="server",
        engine_config=EngineConfig(max_total_sequence_length=4096),
    )
    max_tokens = 256

    with open("./dist/Phi-3-vision-128k-instruct-q4f16_1-MLC/mlc-chat-config.json", "r", encoding="utf-8") as file:
        model_config = json.load(file)

    prompts = [
#        [data.TextData("USER: What is the meaning of life? ASSISTANT:")],
        [
            data.TextData("<|user|>\n"),
            image_data,
            data.TextData("\nWhat is shown in this image?<|end|>\n<|assistant|>\n"),
            #data.TextData("<|image|>\nWhat is shown in this image?<|end|>\n<|assistant|>\n"),
            #data.TextData("<|image_1|>\nWhat is shown in this image?<|end|>\n<|assistant|>\n"),
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


