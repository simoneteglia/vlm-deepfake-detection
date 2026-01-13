import os
import time
import smtplib
import torch
import numpy as np
from tqdm.auto import tqdm
from PIL import Image
from datasets import load_from_disk
from transformers import AutoProcessor, set_seed
from huggingface_hub import snapshot_download
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
import argparse

torch.manual_seed(9999)
set_seed(9999)


MODEL_MAP = {
    "7B": "./Qwen2.5-VL-7B-Instruct",
    "32B": "./Qwen2.5-VL-32B-Instruct",
    "72B": "./Qwen2.5-VL-72B-Instruct",
}

SYSTEM_PROMPT = """
    You are an IQ 150 expert image analyst specialized in detecting AI-generated or digitally manipulated images.

    You must carefully analyze the
      image and produce **explicit chain-of-thought reasoning**, similar in style to:

    - “Okay, let’s analyze the textures first…”
    - “Now I’ll look at the lighting…”
    - “From this detail I can infer…”
    - “Next, I’ll zoom conceptually into the background…”
    - “Given all these observations, I conclude…”

    Your reasoning must be structured, progressive, and connected, showing how each visual clue contributes to your final classification.

    Analyze the image according to the following criteria:

    1. **Texture and Detail Quality**
    - Look for overly smooth skin, waxy or plastic-like surfaces.
    - Check for unnatural noise patterns or smudged or inconsistent fine details (hair, eyes, fabric, surfaces).

    2. **Lighting and Shadows**
    - Check whether lighting direction is consistent across subjects and objects.
    - Inspect reflections in eyes, glasses, metal surfaces, mirrors, tables, or windows.

    3. **Anatomy and Object Structure**
    - Inspect fingers, hands, teeth, limbs, clothing textures, and object shapes.
    - Look for extra, missing, distorted, or fused parts.

    4. **Background Consistency**
    - Look for warped text, repeating patterns, melted geometry, incorrect perspective, and lens inconsistencies.

    5. **Global Coherence**
    - Ask yourself: Does everything in the image behave physically and visually as expected in the real world?
    - Combine all previous steps into a holistic judgment.

    Your task:

    - Provide a **detailed chain-of-thought analysis**, explicitly moving step-by-step through the image.
    - Maintain a natural introspective tone (“Okay, now I’ll examine…”, “This suggests…”, “However, I notice…”).
    - After reasoning in detail, output:

        Answer: REAL or Answer: FAKE
        
    Only one of the two.
"""

PROMPTS = {
    "normal": """
    You are an IQ 150 expert image analyst specialized in detecting AI-generated or digitally manipulated images.

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

    "normal_cot": """
    As you already did perfectly, analyze the image very carefully based on the following criteria, providing detailed chain-of-thought reasoning step-by-step.
    Later, based on your reasoning, provide your judgment using only one word:
    Answer **REAL** if the image is a natural photograph.
    Answer **FAKE** if the image appears AI-generated or manipulated.
        
    Only one of the two.

    Begin your reasoning now:
    """,

    "short": "Is this image real or fake? Choose one: REAL or FAKE. Do not explain your reasoning. Do not output anything except the final classification word."
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

ANSWER_FAKE_COT = """
Analysis: Okay, let's break this image down step by step. First, I’ll focus on the dog's fur texture. At a glance it looks detailed, but when I look closer I notice that the strands don't show natural randomness—there’s a very uniform softness and an almost airbrushed quality. Real fur usually contains tiny irregular shadows, thickness variations, and stray strands, but here everything blends too smoothly.
Now let’s analyze the lighting. The illumination on the dog’s coat is extremely clean and evenly distributed, creating silky gradients rather than the chaotic micro-shadows one expects from dense fur. This suggests a generative lighting model rather than real-world light behavior.
Next, I’ll examine the background. The room behind the dog is blurred, but the blur feels algorithmically perfect—very smooth, with no lens noise, chromatic aberration, or depth granularity. Real optical blur usually contains subtle artifacts depending on the lens. Here it looks artificially consistent.
Now I’ll inspect the object in front of the dog—the controller. Its edges appear slightly too soft, and the surface reflections lack the normal imperfections of real consumer plastics. It looks like a clean synthetic rendering.
Putting all of this together: uniform fur texture, overly idealized lighting, unnatural depth-of-field behavior, and slightly stylized object rendering. These patterns collectively indicate that this is an AI-generated scene rather than a natural photograph.
Answer: FAKE"""

ANSWER_REAL_COT = f"""
Analysis: Okay, let's break this image down step by step. First, I’ll start by examining the people in the image. Their facial features show natural asymmetry, realistic skin texture, and subtle lighting variations. The clothing folds look organic, with irregular wrinkles and material-dependent shading. These details are hard for generative models to reproduce consistently.
Now let’s analyze the lighting. The illumination across the group varies in a believable way: some faces are slightly brighter, others slightly more in shadow, just as you would expect from indoor ambient light. There are no unnaturally smooth gradients or overly consistent highlights that would suggest synthetic rendering. The reflections on the glass table and metallic chair frames also behave exactly as real reflections would—distorted by angle and distance rather than algorithmically smoothed.
Next, I’ll focus on the background. The plaques and framed certificates on the wall show sharp text, proper perspective, and natural variations in shadow and reflection. The calendar features clean print detail without the telltale distortions or warped characters typical of AI-generated text. The whiteboard behind the group has small marks and irregularities that look like they were erased by hand, not artificially generated.
Now I’ll look at the objects in the scene. The notebooks, documents, and pens on the glass table all align correctly with the perspective. Their edges are crisp but not unnaturally perfect. The reflections on the table accurately mirror the shapes and colors of the people standing above it, which is something diffusion models struggle to maintain without inconsistencies.
Putting all of this together: realistic facial texture, natural lighting variability, high-fidelity text, physically accurate reflections, and coherent perspective relationships. All of these features reinforce that this is a genuine photograph taken in a real room with real people rather than a synthetic composition.
Answer: REAL
"""

def load_model(model_size, quantization):
    """Loads the Qwen2.5-VL model with specified quantization."""
    model_name = MODEL_MAP[model_size]

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    if quantization == "none":
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True
        )

    elif quantization == "4bit":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_type=torch.bfloat16
        )

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
        )
    elif quantization == "8bit":
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
        )
    

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded {model_name} | Params: {total_params/1e9:.2f}B")

    # model = model.to(device)
    model.eval()

    return model, processor


def prepare_chat(processor, img, prompt, shot):
    if shot == "zero":
        messages = [
             {"role": "system",
            "content":[
                {"type": "text", "text" : SYSTEM_PROMPT}
            ]},
            {"role": "user",
             "content": [{"type": "image", "image": img},
                         {"type": "text", "text": prompt}]}
        ]
    elif shot == "one":
        messages = [
            {"role": "system",
            "content":[
                {"type": "text", "text" : SYSTEM_PROMPT}
            ]},
            {"role": "user", 
             "content": [
                {"type": "image", "image": FAKE_IMAGE_PATH},
                {"type": "text", "text": prompt}]},
            {"role": "assistant", 
             "content": [{"type": "text", "text": ANSWER_FAKE_COT}]},
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt}]}
        ]
    elif shot == "two":
        messages = [
             {"role": "system",
            "content":[
                {"type": "text", "text" : SYSTEM_PROMPT}
            ]},
            {"role": "user", 
             "content": [
                {"type": "image", "image": FAKE_IMAGE_PATH},
                {"type": "text", "text": PROMPTS['context']}]},
            {"role": "assistant", 
             "content": [{"type": "text", "text": "FAKE"}]},
            {"role": "user", 
             "content": [
                {"type": "image", "image": REAL_IMAGE_PATH},
                {"type": "text", "text": PROMPTS['context']}]},
            {"role": "assistant", 
             "content": [{"type": "text", "text": "REAL"}]},
            {"role": "user", 
             "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt}]}
        ]

    image_inputs, _ =  process_vision_info(messages)
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True), image_inputs


def run(args):
    """Main function to run the inference and evaluation."""
    model_size, dataset_name, shot, prompt_type, quantization = args.model_size, args.dataset_name, args.shot, args.prompt_type, args.quantization
    model, processor = load_model(model_size, quantization)
    dataset = load_from_disk(f"/workspace/dataset/{dataset_name}")
    prompt = PROMPTS[prompt_type]

    preds = []
    real_count = fake_count = 0
    labels = dataset["label"]

    progress = tqdm(dataset, total=len(dataset), desc="Classifying")

    for sample in progress:
        img = sample["image"].convert("RGB")
        chat, images_inputs = prepare_chat(processor, img, prompt, shot)

        #images = [img] if shot == "zero" else [FAKE_IMAGE_PATH, img] if shot == "one" else [FAKE_IMAGE_PATH, REAL_IMAGE_PATH, img] if shot == "two" else [FAKE_IMAGE_PATH, REAL_IMAGE_PATH, TAMPERED_IMAGE_PATH, img]
        inputs = processor(text=[chat], images=images_inputs, padding=True, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=1000, temperature=0.15)

        text = processor.decode(output[0], skip_special_tokens=True).upper()
        answer = text.split("ANSWER:")[-1].strip()

        # print(text)
        # print(f"Predicted answer: {answer}")

        with open("logs.txt", "a") as f:
            f.write("----------------------------------------------------------------------------\n")
            f.write(f"Text:\n{text}\n")
            f.write(f"Answer extracted --->{answer}\n")
            f.write("----------------------------------------------------------------------------\n")


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
    parser.add_argument("-m", "--model_size", type=str, choices=["3B", "7B", "32B", "72B"], required=True, help="Size of the model to use.")
    parser.add_argument("-d", "--dataset_name", type=str, required=True, help="Name of the dataset to use.")
    parser.add_argument("-s", "--shot", type=str, choices=["zero", "one", "two", "three"], required=True, help="Few-shot learning setting.")
    parser.add_argument("-p", "--prompt_type", type=str, choices=["normal", "short", "normal_cot"], required=True, help="Type of prompt to use.")
    parser.add_argument("-q", "--quantization", type=str, choices=["none", "4bit", "8bit"], default="none", help="Quantization method.")
    args = parser.parse_args()

    run(args)