import sys
import json
from mlc_llm import MLCEngine
from mlc_llm.callback import StreamToStdout


def main(device, model_path, lib_path):
    print("device:", device)
    print("Model Path:", model_path)
    print("Library Path:", lib_path)
    # Add your code logic here

    # Create engine
    engine = MLCEngine(model_path, model_lib=lib_path, device=device)

    # Run chat completion in OpenAI API.
    for response in engine.chat.completions.create(
        #messages=[{"role": "user", "content": "Hello?"}],
        messages=[{"role": "user", "content": "What is the meaning of life?"}],
        model=model_path,
        stream=True,
    ):
        for choice in response.choices:
            print(choice.delta.content, end="", flush=True)
    print("\n")
    stats = json.loads(engine.stats())
    #print(f"\n\n\nStatistics: prefill:{stats['total_prefill_tokens']/stats['engine_total_prefill_time']} tok/s, decode:{stats['total_decode_tokens']/stats['engine_total_decode_time']} tok/s\n")
    print(f"\n\n\nStatistics: prefill:{round(stats['total_prefill_tokens']/stats['engine_total_prefill_time'], 1)} tok/s, decode:{round(stats['total_decode_tokens']/stats['engine_total_decode_time'], 1)} tok/s\n")

    engine.terminate()


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script_name.py <device> <model_path> <lib_path>")
        sys.exit(1)
    device = sys.argv[1]
    model_path = sys.argv[2]
    lib_path = sys.argv[3]
    main(device, model_path, lib_path)

