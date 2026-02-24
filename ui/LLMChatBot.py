from unsloth import FastLanguageModel
from transformers import TextIteratorStreamer
from threading import Thread
from dataset.save_cmd_dataset import system_prompt as cmd_system_prompt
from dataset.save_dataset import system_prompt as knowledge_system_prompt
from dataset.save_router_dataset import system_prompt as router_system_prompt
default_system_prompt = "you are a helpful assistant."


# from spm_gpt.save_dataset import system_prompt
router_config = {
    "C": {
        "lora_model_path": None,
        "system_prompt": default_system_prompt,
        "generation_param": {"temperature": 0.8, "top_p": 0.95, "top_k": 50},
    },
    "B": {
        "lora_model_path": "../spm_gpt/finetune/Phi-4_unsloth_spmcmd/checkpoint-348",
        "system_prompt": cmd_system_prompt,
        "generation_param": {"temperature": 0, "do_sample": False}
    },
    "A": {
        "lora_model_path": "../spm_gpt/finetune/Phi-4-unsloth-bnb-4bit_distill_unsloth_results/checkpoint-3024",
        "system_prompt": knowledge_system_prompt,
        "generation_param": {"temperature": 0.5, "top_p": 0.95, "repetition_penalty": 1.0},
    },
}


class LLMChatBot:
    def __init__(self, max_seq_length, model_path="unsloth/Phi-4-unsloth-bnb-4bit", _router_config=None):
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.use_cache = False
        self.model.enable_input_require_grads()


        for name, cfg in router_config.items():
            if cfg["lora_model_path"]:
                self.model.load_adapter(
                    cfg["lora_model_path"],
                    adapter_name=name,
                )
        FastLanguageModel.for_inference(self.model)
        if not _router_config:
            _router_config = router_config

        self.router_config = _router_config
        self.max_history_count = 0

    def router_fn(self, input):
        """
        Use base model to decide route: 'A'(knowledge), 'B'(command), or 'C'(others),
        When encounter the other generation text, then route to C
        """
        self.model.disable_adapters()

        prompt = [
            {"role": "system", "content": router_system_prompt},
            {"role": "user", "content": input},
        ]

        input_ids = self.tokenizer.apply_chat_template(
            prompt,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda")

        output_ids = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=1,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        output = self.tokenizer.decode(
            output_ids[0][input_ids.shape[-1]:],
            skip_special_tokens=True,
        ).strip().replace(" ", "")

        if output not in ("A", "B", "C"):
            return "C"
        else:
            return output



    def _build_input_tokens(self, input, history_list, system_prompt):
        user_msg = {"role": "user", "content": input}

        prompt_text = [
            {"role": "system", "content": system_prompt},
        ]

        if history_list is not None and self.max_history_count > 0:
            for it in history_list[-self.max_history_count:]:
                prompt_text.append(it[0])
                prompt_text.append(it[1])

        prompt_text.append(user_msg)

        inputs = self.tokenizer.apply_chat_template(
            prompt_text,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda")

        return inputs


    def generate_response(self, input, max_new_tokens=128, history_list=None):
        route = self.router_fn(input)
        cfg = self.router_config[route]

        if cfg["lora_model_path"]:
            self.model.enable_adapters()
            self.model.set_adapter(route)
        else:
            self.model.disable_adapters()

        input_ids = self._build_input_tokens(
            input,
            history_list,
            system_prompt=cfg["system_prompt"],
        )

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        generation_kwargs = dict(
            input_ids=input_ids,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            **cfg["generation_param"],
        )


        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        generated_text = ""
        for new_text in streamer:
            generated_text += new_text

        return generated_text, route


    def generate_response_stream(self, input, max_new_tokens=128, history_list=None):
        route = self.router_fn(input)
        cfg = self.router_config[route]

        if cfg["lora_model_path"]:
            self.model.enable_adapters()
            self.model.set_adapter(route)
        else:
            self.model.disable_adapters()

        input_ids = self._build_input_tokens(
            input,
            history_list,
            system_prompt=cfg["system_prompt"],
        )

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        generation_kwargs = dict(
            input_ids=input_ids,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            **cfg["generation_param"],
        )

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        for new_text in streamer:
            yield new_text, route


if __name__ == '__main__':
    llm = LLMChatBot(max_seq_length=4000)
    # print(*llm.generate_response("write python code to process raw STM dI/dV data", max_new_tokens=3000))
    print(*llm.generate_response("scan in 5×5nm with -1V bias"))
    # print(*llm.generate_response("I'm tired of experiment. Tell me a joke to make me happy"))