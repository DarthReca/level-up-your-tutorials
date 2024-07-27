import torch
from dragonfly.models.modeling_dragonfly import DragonflyForCausalLM
from dragonfly.models.processing_dragonfly import DragonflyProcessor
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer


class Dragonfly:
    def __init__(self, use_history: bool = True, cache_dir="cache"):
        device_map = "cuda:0"

        self.use_history = use_history

        self.tokenizer = AutoTokenizer.from_pretrained(
            "togethercomputer/Llama-3-8B-Dragonfly-v1", cache_dir=cache_dir
        )
        image_processor = AutoProcessor.from_pretrained(
            "openai/clip-vit-base-patch32", cache_dir=cache_dir
        ).image_processor
        self.processor = DragonflyProcessor(
            image_processor=image_processor,
            tokenizer=self.tokenizer,
            image_encoding_style="llava-hd",
        )

        self.model = (
            DragonflyForCausalLM.from_pretrained(
                "togethercomputer/Llama-3-8B-Dragonfly-v1", cache_dir=cache_dir
            )
            .to(torch.bfloat16)
            .to(device_map)
        )

        self.history = ""

    def clean_history(self):
        self.history = ""

    def generate(self, image_file, text, input_size=448, max_num=6):
        text = f"<|start_header_id|>user<|end_header_id|>\n\n{text}<|eot_id|>\n\n<|start_header_id|>assistant<|end_header_id|>\n\n"
        inputs = self.processor(
            text=[text],
            images=[self._load_image(image_file)],
            max_length=2048,
            return_tensors="pt",
            is_generate=True,
        ).to(self.model.device)

        with torch.inference_mode():
            generation_output = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                eos_token_id=self.tokenizer.encode("<|eot_id|>"),
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=False,
                temperature=0,
                use_cache=True,
            )

        response = self.processor.batch_decode(
            generation_output, skip_special_tokens=False
        )[0]
        response = self._sanitize(response, skip_special_tokens=False)

        self.history = (
            "\n".join([self.history, text, response]) if self.use_history else None
        )
        return response

    def _load_image(self, filepath: str) -> Image:
        image = Image.open(filepath)
        image = image.convert("RGB").resize((1920, 1080), resample=Image.BILINEAR)
        return image

    def _sanitize(self, s: str, skip_special_tokens: bool) -> str:
        if skip_special_tokens:
            model_reply = s.split("assistant")[-1].strip()
        else:
            model_reply = s.split("<|start_header_id|>assistant<|end_header_id|>")[
                -1
            ].strip()
        model_reply = model_reply.replace("<|reserved_special_token_0|>", "").replace(
            "<|reserved_special_token_1|>", ""
        )
        return model_reply
