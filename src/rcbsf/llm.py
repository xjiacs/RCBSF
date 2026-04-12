import os, torch, json, time
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextStreamer
)

@dataclass
class GenConfig:
    max_new_tokens: int = 3000
    temperature: float = 0.7
    top_p: float = 0.95
    do_sample: bool = True
    seed: int = 42
    stop: Optional[List[str]] = None


class LoggingTextStreamer(TextStreamer):
    def __init__(
        self,
        tokenizer,
        log_file_path: str,
        skip_prompt: bool = True,
        decode_kwargs: Optional[Dict] = None
    ):
        super().__init__(tokenizer, skip_prompt=skip_prompt, decode_kwargs=decode_kwargs)
        self.log_file_path = log_file_path
        self.generated_text = []

    def on_finalized_text(self, text: str, stream_end: bool = False):
        super().on_finalized_text(text, stream_end)
        if text:
            with open(self.log_file_path, "a", encoding="utf-8") as f:
                f.write(text)
            self.generated_text.append(text)


class LocalQwen:
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        log_file_path: Optional[str] = None
    ):
        self.model_path = model_path or os.environ.get("QWEN_MODEL_PATH", "")
        if not self.model_path:
            raise ValueError("Please set QWEN_MODEL_PATH or pass model_path to LocalQwen.")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if log_file_path is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.log_file_path = f"./txt/qwen_chat_log_{timestamp}.txt"
        else:
            self.log_file_path = log_file_path

        if dtype is None:
            dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            use_fast=False
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map="auto" if self.device == "cuda" else None
        )
        if self.device == "cpu":
            self.model.to(self.device)

    def _apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = True
    ) -> Dict[str, Union[torch.Tensor, List[int]]]:
        if hasattr(self.tokenizer, "apply_chat_template"):
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt
            )
            inputs = self.tokenizer([text], return_tensors="pt")
        else:
            concat = ""
            for m in messages:
                role = m.get("role", "user")
                content = m.get("content", "")
                concat += f"<<{role}>>: {content}\n"
            concat += "<<assistant>>:"
            inputs = self.tokenizer([concat], return_tensors="pt")

        if self.device == "cuda":
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        return inputs

    def _write_input_log(self, messages: List[Dict[str, str]], gen_config: GenConfig):
        if not self.log_file_path:
            return
        input_log = f"\n=== Input [{time.strftime('%Y-%m-%d %H:%M:%S')}] ===\n"
        input_log += f"Generation config: {gen_config}\n"
        input_log += f"Messages: {json.dumps(messages, ensure_ascii=False, indent=2)}\n"
        input_log += "=== Model output start ===\n"
        with open(self.log_file_path, "a", encoding="utf-8") as f:
            f.write(input_log)

    def _write_output_end_log(self):
        if not self.log_file_path:
            return
        end_log = "\n=== Model output end ===\n"
        with open(self.log_file_path, "a", encoding="utf-8") as f:
            f.write(end_log)

    def generate(
        self,
        messages: List[Dict[str, str]],
        gen_config: Optional[GenConfig] = None
    ) -> str:
        gen_config = gen_config or GenConfig()

        if gen_config.seed is not None:
            import random, numpy as np
            random.seed(gen_config.seed)
            np.random.seed(gen_config.seed)
            torch.manual_seed(gen_config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(gen_config.seed)

        inputs = self._apply_chat_template(messages)
        self._write_input_log(messages, gen_config)

        eos_token_id = self.tokenizer.eos_token_id
        streamer = None
        if self.log_file_path:
            streamer = LoggingTextStreamer(
                self.tokenizer,
                log_file_path=self.log_file_path,
                skip_prompt=True,
                decode_kwargs={"skip_special_tokens": True}
            )

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=gen_config.max_new_tokens,
                temperature=gen_config.temperature,
                top_p=gen_config.top_p,
                do_sample=gen_config.do_sample,
                eos_token_id=eos_token_id,
                streamer=streamer,
                pad_token_id=self.tokenizer.pad_token_id or eos_token_id
            )

        if streamer:
            text = "".join(streamer.generated_text).strip()
        else:
            text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            if hasattr(self.tokenizer, "apply_chat_template"):
                input_text = self.tokenizer.decode(
                    inputs["input_ids"][0],
                    skip_special_tokens=True
                )
                if text.startswith(input_text):
                    text = text[len(input_text):]
            text = text.strip()

        self._write_output_end_log()
        return text
