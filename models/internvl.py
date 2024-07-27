import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer


class InternVL:
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    VARIANTS = {
        "V1.5": "OpenGVLab/InternVL-Chat-V1-5",
        "V1.5 4B": "OpenGVLab/Mini-InternVL-Chat-4B-V1-5",
        "V2 8B": "OpenGVLab/InternVL2-8B",
        "V2 4B": "OpenGVLab/InternVL2-4B",
        "V2 26B": "OpenGVLab/InternVL2-26B",
    }

    def __init__(self, model_name: str, use_history: bool = True, cache_dir="cache"):
        path = self.VARIANTS[model_name]
        device_map = "cuda:0" if any(x in model_name for x in ["4B", "8B"]) else "auto"

        self.use_history = use_history
        self.tokenizer = AutoTokenizer.from_pretrained(
            path, trust_remote_code=True, cache_dir=cache_dir
        )
        self.model = AutoModel.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            cache_dir=cache_dir,
        ).eval()

        self.generation_config = dict(
            num_beams=1,
            max_new_tokens=512,
            do_sample=False,
        )

        self.history = None

    def clean_history(self):
        self.history = None

    def generate(self, image_file, text, input_size=448, max_num=6):
        pixel_values = (
            self._load_image(image_file, input_size=input_size, max_num=max_num)
            .to(torch.bfloat16)
            .to(self.model.device)
        )
        outputs = self.model.chat(
            self.tokenizer,
            pixel_values,
            text,
            self.generation_config,
            history=self.history,
            return_history=self.use_history,
        )
        if isinstance(outputs, tuple):
            outputs, self.history = outputs
        return outputs

    def _build_transform(self, input_size):
        MEAN, STD = self.IMAGENET_MEAN, self.IMAGENET_STD
        transform = T.Compose(
            [
                T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                T.Resize(
                    (input_size, input_size), interpolation=InterpolationMode.BICUBIC
                ),
                T.ToTensor(),
                T.Normalize(mean=MEAN, std=STD),
            ]
        )
        return transform

    def _find_closest_aspect_ratio(
        self, aspect_ratio, target_ratios, width, height, image_size
    ) -> tuple[float, float]:
        best_ratio_diff = float("inf")
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def _dynamic_preprocess(
        self, image, min_num=1, max_num=6, image_size=448, use_thumbnail=False
    ):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # calculate the existing image aspect ratio
        target_ratios = set(
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= max_num and i * j >= min_num
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        target_aspect_ratio = self._find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size
        )

        # calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size,
            )
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    def _load_image(self, image_file, input_size=448, max_num=6):
        image = (
            Image.open(image_file)
            .convert("RGB")
            .resize((1920, 1080), resample=Image.BILINEAR)
        )
        transform = self._build_transform(input_size=input_size)
        images = self._dynamic_preprocess(
            image, image_size=input_size, use_thumbnail=True, max_num=max_num
        )
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values
