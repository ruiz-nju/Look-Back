import os
import math
from PIL import Image
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from mathruler.grader import extract_boxed_content, grade_answer

def load_image(image_path: str, min_pixels: int, max_pixels: int) -> Image.Image:
    """Load and preprocess an image"""
    try:
        image = Image.open(image_path).convert("RGB")
        
        # Resize if too large or too small
        if (image.width * image.height) > max_pixels:
            resize_factor = math.sqrt(max_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))
        
        if (image.width * image.height) < min_pixels:
            resize_factor = math.sqrt(min_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))
        
        return image
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return None

def prepare_prompts(dataset_name: str, samples: List[Dict], args) -> Tuple[List[Dict], List[Dict]]:
    """Prepare prompts for all samples"""
    prompts = []
    metadata = []
    
    for item in tqdm(samples, desc=f"Preparing {dataset_name} prompts"):
        # Skip if image doesn't exist
        if not os.path.exists(item["image_path"]):
            continue
        
        # Load image
        image = load_image(item["image_path"], args.min_pixels, args.max_pixels)
        if image is None:
            continue
        
        # if "<image>" not in item['question']:
        #     item['question'] = "<image>\n" + item['question']

        # Create prompt
        if args.version == "grpo":
            prompt_text = f"<|im_start|>system\n{args.system_prompt}<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{item['question']}<|im_end|>\n<|im_start|>assistant\n"
        elif args.version == "back":
            prompt_text = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{item['question']} {args.system_prompt}<|im_end|>\n<|im_start|>assistant\n"
        elif args.version == "hint":
            prompt_text = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|> {args.system_prompt} \n Qwestion: {item['question']}<|im_end|>\n<|im_start|>assistant\n"
        else:
            raise
        
        prompts.append({
            "prompt": prompt_text,
            "multi_modal_data": {"image": image},
        })
        
        metadata.append({
            "dataset": dataset_name,
            "id": item["id"],
            "question": item["question"],
            "answer": item["answer"],
            "prompt": prompt_text,
            **{k: v for k, v in item.items() if k not in ["image_path", "dataset", "id", "question", "answer"]}
        })
    
    return prompts, metadata

def process_outputs_simplified(outputs, metadata) -> List[Dict]:
    results = []
    for i, output in enumerate(outputs):
        prediction = output.outputs[0].text.strip()
        meta = metadata[i]
        
        result = {
            "id": meta["id"],
            "question": meta["question"],
            "answer": meta["answer"],
            "prediction": prediction
        }
        results.append(result)
    
    return results
