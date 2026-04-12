import os, torch, json, sys, time
from typing import List, Dict, Optional, Union
from dataclasses import dataclass, asdict
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig as HFGenConfig, TextStreamer

try:
    from llama_cpp import Llama
except Exception:
    Llama = None


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


@dataclass
class GenConfig:
    max_new_tokens: int = 3000
    temperature: float = 0.7
    top_p: float = 0.95
    do_sample: bool = True
    stop: Optional[List[str]] = None
    seed: int = 42

    def to_hf_dict(self):
        d = asdict(self)
        d.pop('stop', None)
        d.pop('seed', None)
        return d

    def to_llamacpp_dict(self):
        return {
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stop": self.stop or [],
        }

    def to_chatglm_dict(self):
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }

    def to_baichuan_dict(self):
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }


class BaseModel:
    def __init__(self, model_path: str, log_file_path: Optional[str] = None):
        self.model_path = model_path

        if log_file_path is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.log_file_path = f"./txt/model_chat_log_{timestamp}.txt"
        else:
            self.log_file_path = log_file_path

        if not os.path.exists(model_path):
            print(f"Error: Model path does not exist: {model_path}", file=sys.stderr)
        if self.log_file_path:
            log_dir = os.path.dirname(self.log_file_path)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)

    def _write_input_log(self, messages: List[Dict[str, str]], gen_config: GenConfig):
        if not self.log_file_path:
            return
        input_log = f"\n=== Conversation Input [{time.strftime('%Y-%m-%d %H:%M:%S')}] ===\n"
        input_log += f"Generation Config: {gen_config}\n"
        input_log += f"Conversation Messages: {json.dumps(messages, ensure_ascii=False, indent=2)}\n"
        input_log += "=== Model Output Start ===\n"
        with open(self.log_file_path, "a", encoding="utf-8") as f:
            f.write(input_log)

    def _write_output_end_log(self):
        if not self.log_file_path:
            return
        end_log = "\n=== Model Output End ===\n"
        with open(self.log_file_path, "a", encoding="utf-8") as f:
            f.write(end_log)

    def generate(self, messages: List[Dict[str, str]], gen_config: Optional[GenConfig] = None) -> str:
        raise NotImplementedError


class HuggingFaceBaseModel(BaseModel):
    def __init__(self, model_path: str, log_file_path: Optional[str] = None):
        super().__init__(model_path, log_file_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.device == "cuda":
            if torch.cuda.is_bf16_supported():
                self.dtype = torch.bfloat16
            else:
                self.dtype = torch.float16
        else:
            self.dtype = torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=self.dtype,
            device_map="auto" if self.device == "cuda" else None
        )

        if self.device == "cpu":
            self.model.to(self.device, dtype=self.dtype)

        if getattr(self.tokenizer, "pad_token", None) is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _apply_chat_template(self, messages: List[Dict[str, str]], add_generation_prompt: bool = True) -> str:
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt
            )
        except Exception:
            concat = ""
            for m in messages:
                role = m.get("role", "user")
                content = m.get("content", "")
                concat += f"<<{role}>>: {content}\n"
            concat += "<<assistant>>:"
            return concat

    def generate(self, messages: List[Dict[str, str]], gen_config: Optional[GenConfig] = None) -> str:
        gen_config = gen_config or GenConfig()
        if gen_config.seed is not None:
            torch.manual_seed(gen_config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(gen_config.seed)

        self._write_input_log(messages, gen_config)

        prompt = self._apply_chat_template(messages)
        inputs = self.tokenizer([prompt], return_tensors="pt")
        if self.device == "cuda":
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        gen_dict = gen_config.to_hf_dict()

        eos_token_id = [self.tokenizer.eos_token_id]
        if gen_config.stop:
            try:
                stop_ids = self.tokenizer.encode(gen_config.stop[0], add_special_tokens=False)
                if stop_ids:
                    eos_token_id.append(stop_ids[-1])
            except Exception:
                pass

        streamer = None
        if self.log_file_path:
            streamer = LoggingTextStreamer(
                self.tokenizer,
                log_file_path=self.log_file_path,
                skip_prompt=True,
                decode_kwargs={"skip_special_tokens": True}
            )
            gen_dict["streamer"] = streamer

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                **gen_dict,
                eos_token_id=eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id
            )[0]

        if streamer:
            text = "".join(streamer.generated_text).strip()
        else:
            input_ids_len = inputs["input_ids"].shape[1]
            new_tokens = output_ids[input_ids_len:]
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            text = text.strip()

        self._write_output_end_log()

        return text


class GptOssModel(HuggingFaceBaseModel):
    def __init__(self, model_path: str, log_file_path: Optional[str] = None):
        super().__init__(model_path, log_file_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.im_start_id = self.tokenizer.convert_tokens_to_ids(
            "<|im_start|>") if "<|im_start|>" in self.tokenizer.get_vocab() else None
        self.im_end_id = self.tokenizer.convert_tokens_to_ids(
            "<|im_end|>") if "<|im_end|>" in self.tokenizer.get_vocab() else None

    def _apply_chat_template(self, messages: List[Dict[str, str]], add_generation_prompt: bool = True) -> str:
        prompt = ""

        system_msg = next((m for m in messages if m['role'] == 'system'), None)
        other_msgs = [m for m in messages if m['role'] != 'system']

        json_instruction = (
            "\nYou are a strict JSON generator. Do not explain. Do not analyze in plain text. "
            "Output ONLY a valid JSON object strictly following this format:\n"
            "{\n"
            '  "seed_items": [\n'
            '    { "id": 0, "category": "string", "basis": "string" }\n'
            "  ]\n"
            "}"
        )

        if system_msg:
            content = system_msg['content'] + json_instruction
        else:
            content = "You are a helpful assistant." + json_instruction

        prompt += f"<|im_start|>system\n{content}<|im_end|>\n"

        for msg in other_msgs:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role not in ['user', 'assistant']: role = 'user'
            prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"

        if add_generation_prompt:
            prompt += "<|im_start|>assistant\n```json\n"

        return prompt

    def generate(self, messages: List[Dict[str, str]], gen_config: Optional['GenConfig'] = None) -> str:
        gen_config = gen_config or GenConfig()

        self._write_input_log(messages, gen_config)

        if gen_config.max_new_tokens is None or gen_config.max_new_tokens < 3000:
            gen_config.max_new_tokens = 5000

        prompt = self._apply_chat_template(messages)

        inputs = self.tokenizer([prompt], return_tensors="pt")
        if self.device == "cuda":
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        eos_token_ids = [self.tokenizer.eos_token_id]
        if self.im_end_id: eos_token_ids.append(self.im_end_id)

        streamer = None
        if self.log_file_path:
            streamer = LoggingTextStreamer(
                self.tokenizer,
                log_file_path=self.log_file_path,
                skip_prompt=True,
                decode_kwargs={"skip_special_tokens": True}
            )

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                **gen_config.to_hf_dict(),
                eos_token_id=eos_token_ids,
                pad_token_id=self.tokenizer.pad_token_id,
                streamer=streamer
            )[0]

        if streamer:
            generated_text = "".join(streamer.generated_text).strip()
        else:
            input_len = inputs["input_ids"].shape[1]
            new_tokens = output_ids[input_len:]
            generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        full_response = generated_text.strip()

        if full_response.endswith("```"):
            full_response = full_response[:-3]
        full_response = full_response.strip()

        self._write_output_end_log()

        return full_response


class QwenModel(HuggingFaceBaseModel):
    def __init__(self, model_path: str, log_file_path: Optional[str] = None):
        super().__init__(model_path, log_file_path)

    pass


class LlamaChatModel(HuggingFaceBaseModel):
    def __init__(self, model_path: str, log_file_path: Optional[str] = None):
        super().__init__(model_path, log_file_path)

    def _apply_chat_template(self, messages: List[Dict[str, str]], add_generation_prompt: bool = True) -> str:
        BOT_TEXT = "<|begin_of_text|>"
        HEADER_START = "<|start_header_id|>"
        HEADER_END = "<|end_header_id|>"
        EOT = "<|eot_id|>"

        prompt = BOT_TEXT
        for msg in messages:
            role = msg['role'].lower()
            content = msg['content'].strip()
            prompt += f"{HEADER_START}{role}{HEADER_END}\n{content}{EOT}"

        if add_generation_prompt and messages and messages[-1]['role'].lower() == 'user':
            prompt += f"{HEADER_START}assistant{HEADER_END}\n"

        return prompt


class MistralChatModel(HuggingFaceBaseModel):
    def __init__(self, model_path: str, log_file_path: Optional[str] = None):
        super().__init__(model_path, log_file_path)

    pass


class GemmaChatModel(HuggingFaceBaseModel):
    def __init__(self, model_path: str, log_file_path: Optional[str] = None):
        super().__init__(model_path, log_file_path)

    pass


class GenericModel(HuggingFaceBaseModel):
    def __init__(self, model_path: str, log_file_path: Optional[str] = None):
        super().__init__(model_path, log_file_path)

    pass


class ChatGLMModel(BaseModel):
    def __init__(self, model_path: str, log_file_path: Optional[str] = None):
        super().__init__(model_path, log_file_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
        ).to(self.device).eval()

    def generate(self, messages: List[Dict[str, str]], gen_config: Optional[GenConfig] = None) -> str:
        gen_config = gen_config or GenConfig()
        gen_dict = gen_config.to_chatglm_dict()

        self._write_input_log(messages, gen_config)

        history = []
        query = ""
        system_prompt = ""

        if messages[0]['role'] == 'system':
            if hasattr(self.tokenizer, 'build_chat_input'):
                system_prompt = messages[0]['content']
            messages = messages[1:]

        for msg in messages:
            if msg['role'] == 'user':
                if query:
                    history.append((query, ""))
                query = msg['content']
            elif msg['role'] == 'assistant':
                history.append((query, msg['content']))
                query = ""

        if not query:
            query = "..."

        if hasattr(self.tokenizer, 'build_chat_input'):
            response, _ = self.model.chat(self.tokenizer, query, history=history, system=system_prompt, **gen_dict)
        else:
            response, _ = self.model.chat(self.tokenizer, query, history=history, **gen_dict)
        response = response.strip()

        if self.log_file_path:
            with open(self.log_file_path, "a", encoding="utf-8") as f:
                f.write(response)

        self._write_output_end_log()

        return response


class BaichuanChatModel(BaseModel):
    def __init__(self, model_path: str, log_file_path: Optional[str] = None):
        super().__init__(model_path, log_file_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map="auto"
        ).eval()
        self.model.generation_config = HFGenConfig.from_pretrained(model_path)

    def generate(self, messages: List[Dict[str, str]], gen_config: Optional[GenConfig] = None) -> str:
        gen_config = gen_config or GenConfig()
        gen_dict = gen_config.to_baichuan_dict()

        self._write_input_log(messages, gen_config)

        response = self.model.chat(self.tokenizer, messages, **gen_dict)
        response = response.strip()

        if self.log_file_path:
            with open(self.log_file_path, "a", encoding="utf-8") as f:
                f.write(response)

        self._write_output_end_log()

        return response


class LlamaCppModel(BaseModel):
    def __init__(self, model_path: str, log_file_path: Optional[str] = None):
        super().__init__(model_path, log_file_path)
        if Llama is None:
            raise RuntimeError("error")

        self.llm = Llama(model_path=self.model_path, n_ctx=4096, n_gpu_layers=-1)

    def _apply_chat_template(self, messages: List[Dict[str, str]]) -> str:
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

        prompt = ""
        if messages[0]['role'] == 'system':
            prompt += B_INST + B_SYS + messages[0]['content'] + E_SYS
            messages = messages[1:]

        for i, msg in enumerate(messages):
            is_user = (msg['role'] == 'user')
            if is_user:
                if i == 0 and not prompt:
                    prompt += B_INST + " " + msg['content'] + " "
                else:
                    prompt += B_INST + " " + msg['content'] + " "
            else:
                prompt += E_INST + " " + msg['content'] + " "

        if messages[-1]['role'] == 'user':
            prompt += E_INST

        return prompt

    def generate(self, messages: List[Dict[str, str]], gen_config: Optional[GenConfig] = None) -> str:
        gen_config = gen_config or GenConfig()
        gen_dict = gen_config.to_llamacpp_dict()

        self._write_input_log(messages, gen_config)

        prompt = self._apply_chat_template(messages)
        res = self.llm.create_completion(prompt=prompt, **gen_dict)
        text = res["choices"][0]["text"].strip()

        if self.log_file_path:
            with open(self.log_file_path, "a", encoding="utf-8") as f:
                f.write(text)

        self._write_output_end_log()

        return text