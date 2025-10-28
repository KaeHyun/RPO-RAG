from unsloth import FastLanguageModel
import torch
from datasets import load_dataset, Dataset
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments
from argparse import ArgumentParser
import re
import json


dtype = None
load_in_4bit = True


fourbit_models = [
    "unsloth/mistral-7b-bnb-4bit",
    "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
    "unsloth/llama-2-7b-bnb-4bit",
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
    "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
    "unsloth/llama-2-7b-chat-bnb-4bit",
]


def train(args):
    # Directly read the JSONL file and create a Dataset
    data = []
    with open(args.train_data, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # Ignore empty lines
                data.append(json.loads(line.strip()))
    
    train_dataset = Dataset.from_list(data)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "your_model_path_or_name",
        max_seq_length = args.max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    

    # 1) Response start template (after tokenizer is loaded!)

    response_template = tokenizer.apply_chat_template(
        [{"role": "assistant", "content": ""}],
        tokenize=False,
        add_generation_prompt=False,
    ).lstrip()

    

    # -------- Preview: use original prompt/completion before mapping --------

    FastLanguageModel.for_inference(model)

    last = train_dataset[len(train_dataset) - 1]

    # 1) First, build the input string

    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": last["prompt"]}],
        tokenize=False,
        add_generation_prompt=True,
    )



    # 2) Tokenize the string; tokenizer(...) always returns a dict

    inputs = tokenizer(prompt_text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
    print(tokenizer.batch_decode(outputs)[0])
    print(last["completion"])

    FastLanguageModel.for_training(model)

    # --------------------------------------------------------------------------



    # 3) Dataset preprocessing (convert prompt + completion → single text)
    def normalize_bracket_list(s: str) -> str:
        s = s.strip()
        if not (s.startswith("[") and s.endswith("]")):
            # Wrap data in brackets if missing
            s = f"[{s}]"
        inner = s[1:-1].strip()
        # Split and normalize whitespace
        items = [re.sub(r"\s+", " ", x.strip()) for x in inner.split(",") if x.strip()]
        # Sort alphabetically (useful for order-insensitive evaluation)
        items = sorted(items)
        # 중복 제거
        seen, out = set(), []
        for it in items:
            if it.lower() not in seen:
                seen.add(it.lower())
                out.append(it)
        return f"[{', '.join(out)}]"

    def build_text(example):
        system_msg = {
            "role": "system",
            "content": """Based on the Answer-Centered-Paths, please answer the given question.  
            The Answer-Centered-Paths helps you to step-by-step reasoning to answer the question.  

            Let's think step by step. Return the most possible answers based on the given paths by listing each answer on a separate line.  
            Please keep the answer as simple as possible and return all the possible answers as a list."""
        }
        gold = normalize_bracket_list(example["completion"])

        text = tokenizer.apply_chat_template(
            [
                system_msg,
                {"role": "user", "content": example["prompt"]},
                {"role": "assistant", "content": gold},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )

        return {"text": text}

    train_dataset = train_dataset.map(build_text, remove_columns=train_dataset.column_names)


    # model = FastLanguageModel.get_peft_model(
    #     model,
    #     r = 16,
    #     target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    #     lora_alpha = 16,
    #     lora_dropout = 0,
    #     bias = "none",
    #     use_gradient_checkpointing = "unsloth",
    #     random_state = 42,
    #     use_rslora = False,
    #     loftq_config = None,
    # )

    sft_config = SFTConfig(

        per_device_train_batch_size = 2,
        per_device_eval_batch_size = 1,
        gradient_accumulation_steps = 8,
        warmup_ratio = 0.03,
        num_train_epochs = args.num_epochs,
        learning_rate = args.lr,
        save_total_limit = 3,
        save_steps = 100,
        logging_steps = 25,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        optim = "adamw_8bit",
        weight_decay = 0.0,
        lr_scheduler_type = "cosine",
        seed = 42,
        output_dir = args.output_dir,
        # 🔑 Compute loss only on the answer portion
        completion_only_loss = True,
    )

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,   # Now the dataset only has a "text" column
        dataset_text_field = "text",
        packing = False,
        args = sft_config,
        response_template = response_template,
    )


    trainer.train(resume_from_checkpoint = args.resume_from_checkpoint)
    model.save_pretrained(f"{args.output_dir}/lora_model")



if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument("--llm", default="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit")
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--lr", type=float, default= 2e-4)
    parser.add_argument("--num_epochs", type=int, default= 3)
    parser.add_argument("--output_dir", default="write_your_output_directory")
    parser.add_argument("--train_data", default= "user_train_data.jsonl")
    parser.add_argument("--resume_from_checkpoint", action='store_true')

    args = parser.parse_args()

    train(args)
