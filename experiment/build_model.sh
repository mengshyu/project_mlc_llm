#!/bin/bash
#set -e
source ~/workspace/venv/bin/activate

ROOT=$PWD

#QUAN_PARAMS=q0f32
#QUAN_PARAMS=q0f16
QUAN_PARAMS=q4f16_1
#QUAN_PARAMS=q4f16_ft
DEVICE=opencl

function Clean()
{
    rm -rf $output_dir
    mkdir $output_dir
}

function ConvertWeight()
{
    mlc_llm convert_weight ./models/$model/ \
        --quantization $QUAN_PARAMS \
        -o $output_dir
}

function GenConfig()
{
    mlc_llm gen_config ./models/$model/ \
        --quantization $QUAN_PARAMS \
        --conv-template $conv_temp \
        -o $output_dir
}

function Compile()
{
    mlc_llm compile \
    	$output_dir/mlc-chat-config.json \
    	--device $DEVICE \
        -o $model_lib
}

function Benchmark()
{
    mlc_llm bench \
        $output_dir \
        --model-lib $model_lib  \
        --device $DEVICE \
        --prompt "what is the meaning of life?" \
        --generate-length 128


}

# Define the array of dictionaries

declare -A dictionaries=(
#    ["gorilla"]="gorilla-openfunctions-v2"
#    ["redpajama_chat"]="RedPajama-INCITE-Chat-3B-v1"
    ["phi-2"]="phi-2"
#    ["llama-2"]="Llama-2-7b-chat-hf"
#    ["mistral_default"]="Mistral-7B-Instruct-v0.2"
#    ["chatml"]="Qwen-1_8B-Chat"
#    ["chatml"]="Qwen-7B-Chat"
#    ["llama_default"]="TinyLlama-1.1B-Chat-v0.4"
#    ["gemma_instruction"]="gemma-2b-it"
#    ["gpt2"]="gpt2"
#     ["llama-3"]="Llama-3-8B-Instruct"
#    ["phi-3"]="Phi-3-mini-4k-instruct"
#    ["phi-3"]="Phi-3-mini-128k-instruct"
)



# Iterate over each dictionary
for conv_temp in "${!dictionaries[@]}"; do
    cd $ROOT

    model="${dictionaries[$conv_temp]}"
    echo "conv temp:$conv_temp model:$model"

    output_dir=$PWD/dist/$model-$QUAN_PARAMS-MLC
    model_lib=$output_dir/$model-$QUAN_PARAMS-mali.so

    #Clean
    #ConvertWeight
    GenConfig
    Compile
    Benchmark

    #python chat.py $DEVICE $output_dir $model_lib

    read -n 1 -s -r -p "Press any key to continue..."
    echo  # Add a newline for better readability

done

