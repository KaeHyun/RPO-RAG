from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import os
from .base_language_model import BaseLanguageModel
from transformers import LlamaTokenizer
from peft import PeftModel
import json
from peft import AutoPeftModelForCausalLM

class Llama(BaseLanguageModel):
    DTYPE = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    HF_TOKEN = "hf_xxxxxxx"  # Replace with your actual Hugging Face token
    @staticmethod
    def add_args(parser):
        parser.add_argument('--model_path', type=str, help="HUGGING FACE MODEL or model path", default='meta-llama/Llama-2-7b-chat-hf')
        parser.add_argument('--max_new_tokens', type=int, help="max length", default=512)
        parser.add_argument('--dtype', choices=['fp32', 'fp16', 'bf16'], default='fp16')


    def __init__(self, args):
        self.args = args
        self.maximun_token = 4096 - 100
        self.tokenizer = None
        
    def load_model(self, **kwargs):
        model = LlamaTokenizer.from_pretrained(**kwargs, use_fast=False, token=self.HF_TOKEN)
        return model
    
    def tokenize(self, text):
        if self.tokenizer is not None:
            return len(self.tokenizer.tokenize(text))
        else:
            # Fallback: create a temporary tokenizer for tokenization
            # Check if it's a LoRA model to get the correct base model
            is_lora = os.path.exists(self.args.model_path + "/adapter_config.json")
            if is_lora:
                import json
                with open(os.path.join(self.args.model_path, "adapter_config.json"), "r") as f:
                    adapter_config = json.load(f)
                base_model_name = adapter_config.get("base_model_name_or_path", "meta-llama/Llama-2-7b-chat-hf")
                temp_tokenizer = AutoTokenizer.from_pretrained(
                    base_model_name, 
                    use_fast=False, 
                    token=self.HF_TOKEN
                )
            else:
                temp_tokenizer = AutoTokenizer.from_pretrained(
                    self.args.model_path, 
                    use_fast=False, 
                    token=self.HF_TOKEN
                )
            return len(temp_tokenizer.tokenize(text))
    
    def prepare_for_inference(self, **model_kwargs):

        adapter_cfg = os.path.join(self.args.model_path, "adapter_config.json")
        is_lora = os.path.exists(adapter_cfg)

        if is_lora: 
            with open(adapter_cfg, "r") as f:
                base_model_name = json.load(f).get("base_model_name_or_path")
        else:
            base_model_name = self.args.model_path

        print(f"Base Model: {base_model_name}")
        
        # 항상 베이스 모델에서 토크나이저 로드하기 
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            use_fast = False, 
            token=self.HF_TOKEN
        )

        if self.tokenizer.pad_token is None: 
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.tokenizer.padding_side="left" 
        # Check if it's a LoRA model by looking for adapter_config.json
        # is_lora = os.path.exists(self.args.model_path + "/adapter_config.json")
        
        if is_lora:
            print("Loading LoRA model from:", self.args.model_path)
            model = AutoPeftModelForCausalLM.from_pretrained(
                self.args.model_path, 
                device_map="auto", 
                torch_dtype=self.DTYPE.get(self.args.dtype, torch.bfloat16)
            )

            model = model.merge_and_unload()
        else:
            print("Loading HuggingFace model from:", self.args.model_path)
            model = AutoModelForCausalLM.from_pretrained(
                self.args.model_path, 
                device_map="auto", 
                torch_dtype=self.DTYPE.get(self.args.dtype, torch.bfloat16),
                **model_kwargs
            )

        #     from peft import AutoPeftModelForCausalLM
        #     import json
            
        #     # Load adapter config to get base model
        #     with open(os.path.join(self.args.model_path, "adapter_config.json"), "r") as f:
        #         adapter_config = json.load(f)
        #     base_model_name = adapter_config.get("base_model_name_or_path", "meta-llama/Llama-2-7b-chat-hf")
            
        #     # Load tokenizer - try multiple approaches for LoRA models
        #     print(f"Base model: {base_model_name}")
            
        #     # First try: Load from LoRA model directory (if it has tokenizer files)
        #     try:
        #         print("Trying to load tokenizer from LoRA model directory...")
        #         self.tokenizer = AutoTokenizer.from_pretrained(
        #             self.args.model_path,
        #             use_fast=False,
        #             token=self.HF_TOKEN
        #         )
        #         print(f"Tokenizer loaded from LoRA directory: {type(self.tokenizer)}")
        #     except Exception as e:
        #         print(f"Failed to load from LoRA directory: {e}")
                
        #         # Second try: Use standard Llama-3.1-8B-Instruct (not the 4bit version)
        #         try:
        #             print("Trying standard Llama-3.1-8B-Instruct tokenizer...")
        #             self.tokenizer = AutoTokenizer.from_pretrained(
        #                 "meta-llama/Llama-2-7b-chat-hf",
        #                 use_fast=False,
        #                 token=self.HF_TOKEN
        #             )
        #             print(f"Tokenizer loaded from standard model: {type(self.tokenizer)}")
        #         except Exception as e2:
        #             print(f"Failed to load standard tokenizer: {e2}")
                    
        #             # Third try: Use Llama-2 tokenizer as last resort
        #             print("Trying Llama-2 tokenizer as fallback...")
        #             self.tokenizer = AutoTokenizer.from_pretrained(
        #                 "meta-llama/Llama-2-7b-chat-hf",
        #                 use_fast=False,
        #                 token=self.HF_TOKEN
        #             )
        #             print(f"Tokenizer loaded from Llama-2: {type(self.tokenizer)}")
            
            # # Verify tokenizer is valid
            # if not hasattr(self.tokenizer, 'pad_token') or self.tokenizer is None or isinstance(self.tokenizer, bool):
            #     print("ERROR: Tokenizer is not valid!")
            #     raise ValueError("Failed to load a valid tokenizer")
            
            # # Set pad token if not set
            # if self.tokenizer.pad_token is None:
            #     self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # # Load LoRA model
            # model = AutoPeftModelForCausalLM.from_pretrained(
            #     self.args.model_path, 
            #     device_map="auto", 
            #     torch_dtype=self.DTYPE.get(self.args.dtype, torch.bfloat16)
            # )

            # model = model.merge_and_unload()
            
        # Create pipeline with LoRA model
        self.generator = pipeline(
            "text-generation", 
            model=model, 
            tokenizer=self.tokenizer, 
            device_map="auto", 
            model_kwargs=model_kwargs, 
            torch_dtype=self.DTYPE.get(self.args.dtype, torch.bfloat16)
        )

    @torch.inference_mode()
    def generate_sentence(self, llm_input):
        outputs = self.generator(llm_input, return_full_text=False, max_new_tokens=self.args.max_new_tokens)
        return outputs[0]['generated_text'] # type: ignore