import math
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from transformers import AutoProcessor, set_seed, AutoConfig
import os 
from datasets import load_from_disk
from tqdm import tqdm
import argparse

torch.manual_seed(9999)
set_seed(9999)

MODEL_MAP = {
    "8B": "./InternVL3_5-8B-Instruct",
    "38B": "./InternVL3_5-38B-Instruct"
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

PROMPTS = {
    "normal": """<image/>\nYou are an IQ 150 expert image analyst specialized in detecting AI-generated or digitally manipulated images.

                  Analyze the image very carefully based on the following criteria:

                  1. **Texture and Detail Quality**
                    - Look for overly smooth skin, waxy or plastic-like surfaces.
                    - Check for unnatural noise patterns or smudged fine details (ears, eyes, jewelry, hair).

                  2. **Lighting and Shadows**
                    - Verify that all objects follow the same light direction.
                    - Check reflections in eyes, glasses, mirrors, metal objects, and water.

                  3. **Anatomy and Object Structure**
                    - Inspect fingers, hands, teeth, limbs, clothing textures, and object shapes.
                    - Look for extra, missing, distorted, or fused parts.

                  4. **Background Consistency**
                    - Check for warped text, repeating patterns, melted edges, or implausible scene geometry.

                  5. **Global Coherence**
                    - Ask: Does everything in the image visually make sense in the real world?

                  After analyzing the image **internally**, provide your judgment using **only one word**:

                  - Output **REAL** if the image is a natural photograph.
                  - Output **FAKE** if the image appears AI-generated or manipulated.

                  Answer:""",

    "short": "<image/>\nIs this image real or fake? Choose one: REAL or FAKE. Do not explain your reasoning. Do not output anything except the final classification word."
}

FAKE_IMAGE_PATH = "./test_images/fake_img3.jpg"
REAL_IMAGE_PATH = "./test_images/real_img2.jpg"
TAMPERED_IMAGE_PATH = "./test_images/tampered_img.jpg"

ANSWER_FAKE = """Analysis: The fur texture is overly uniform and has a soft, airbrushed look without the subtle strand-level randomness of real dog fur. The lighting is extremely clean and consistent across the entire coat, lacking the typical micro-shadows real fur generates. The depth of field looks artificially smooth, with the background blur and bed fabric appearing almost too perfectly gradient. The controller has slightly softened edges and an unrealistically pristine surface. Altogether, these signs point to an AI-generated scene rather than a natural photograph.
Answer: FAKE"""
ANSWER_REAL = """Analysis: The people in the image show natural variation in facial texture, lighting, and posture. Their skin tones reflect the warm indoor lighting realistically, with natural shadows and highlights. Clothing exhibits believable fabric folds and imperfections. The reflections on the glass table and the mixed light sources look physically consistent. Background elements—certificates, calendar, whiteboard—show normal photographic noise, depth, and perspective without the softened, airbrushed, or overly-uniform qualities typical of AI-generated images. No warped limbs, smoothed skin, or inconsistent lighting cues are present.
Answer: REAL"""
ANSWER_TAMPERED = """Analysis: The man’s face and hair have a smooth, plastic-like texture with almost no natural skin variation. The beard edges look uniformly sharp and lack the irregular transitions typical of real facial hair. His ears and jawline blend unnaturally into the surrounding skin, and the lighting on his face doesn’t fully match the warmer ambient lighting of the room. The fingers are slightly distorted, with inconsistent proportions and an odd grip on the object. Meanwhile, the background—walls, door panels, plant, and lighting—shows natural photographic noise, realistic depth, and believable lighting falloff. This mismatch between a real photographic environment and an overly clean, stylized subject strongly indicates an AI-generated person composited into a real photo.
Answer: FAKE"""

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
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

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

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
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = image_file.convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    num_layers = config.llm_config.num_hidden_layers
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map

# If you set `load_in_8bit=True`, you will need two 80GB GPUs.
# If you set `load_in_8bit=False`, you will need at least three 80GB GPUs.

def load_model(model_size, quantization):
    """Loads the InternVL model with specified quantization."""
    model_name = MODEL_MAP[model_size]

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
    device_map = split_model(model_name.split("/")[-1])

    if quantization == "none":
       model = AutoModel.from_pretrained(
           model_name,
           dtype=torch.bfloat16,
           load_in_8bit=False,
           low_cpu_mem_usage=True,
           use_flash_attn=True,
           trust_remote_code=True,
           device_map=device_map).eval()
    elif quantization == "8bit":
        model = AutoModel.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            load_in_8bit=True,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map=device_map).eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded {model_name} | Params: {total_params/1e9:.2f}B")

    return model, tokenizer


def prepare_question(prompt_type, shot, sample):

    img = sample["image"]
    pixel_values1 = load_image(img, max_num=12).to(torch.bfloat16).cuda()

    if shot == "zero":
        question = PROMPTS[prompt_type]
        pixel_values = pixel_values1
    elif shot == "one":
        example_img = Image.open(REAL_IMAGE_PATH).convert("RGB")
        pixel_values0 = load_image(example_img, max_num=12).to(torch.bfloat16).cuda()
        question = f"{PROMPTS[prompt_type]}\nAssistant: REAL\n{PROMPTS[prompt_type]}"
        pixel_values = torch.cat((pixel_values0, pixel_values1), dim=0)

    return question, pixel_values



def run(args):
    model_size, dataset_name, shot, prompt_type, quantization = args.model_size, args.dataset_name, args.shot, args.prompt_type, args.quantization

    model, tokenizer = load_model(model_size, quantization)
    dataset = load_from_disk(f"/workspace/dataset/{dataset_name}")
    preds = []
    real_count = fake_count = 0
    labels = dataset["label"]

    progress = tqdm(dataset, total=len(dataset), desc="Classifying")

    for sample in progress:
        question, pixel_values = prepare_question(prompt_type, shot, sample)

        # img = sample["image"]
        # pixel_values = load_image(img, max_num=12).to(torch.bfloat16).cuda()
        # question = PROMPTS[prompt_type]        

        generation_config = dict(max_new_tokens=10, do_sample=False)
        response = model.chat(tokenizer, pixel_values, question, generation_config=generation_config)

        answer = response
        print(f"Predicted answer: {answer}")

        if "REAL" in answer:
            preds.append(0)
            real_count += 1
        elif "FAKE" in answer:
            preds.append(1)
            fake_count += 1
        elif "TAMPERED" in answer:
            preds.append(1)
            fake_count += 1
        else:
            preds.append(-1)

        progress.set_description(f"REAL: {real_count} | FAKE: {fake_count}")

    accuracy = (np.array(preds) == np.array(labels)).mean()

    result_summary = f"""
    Model: {MODEL_MAP[model_size]}
    # Shot: {shot}
    Dataset: {dataset_name}
    Prompt: {prompt_type}
    Quantization: {quantization}
    Accuracy: {accuracy*100:.2f}%
    Real: {real_count}, Fake: {fake_count}
    """
    print(result_summary)

    with open("results.txt", "a") as f:
        f.write(result_summary + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model and dataset arguments
    parser.add_argument("-m", "--model_size", type=str, choices=["8B", "38B"], required=True, help="Size of the model to use.")
    parser.add_argument("-d", "--dataset_name", type=str, required=True, help="Name of the dataset to use.")
    parser.add_argument("-s", "--shot", type=str, choices=["zero", "one", "two", "three"], required=True, help="Few-shot learning setting.")
    parser.add_argument("-p", "--prompt_type", type=str, choices=["normal", "short"], required=True, help="Type of prompt to use.")
    parser.add_argument("-q", "--quantization", type=str, choices=["none", "4bit", "8bit"], default="none", help="Quantization method.")
    args = parser.parse_args()

    run(args)