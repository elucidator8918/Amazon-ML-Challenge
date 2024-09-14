import os
import pandas as pd
import numpy as np
import torch
import warnings
from tqdm.auto import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from constants import entity_unit_map

warnings.filterwarnings("ignore")
tqdm.pandas()

model_id = "Qwen/Qwen2-VL-2B-Instruct"
DATASET_FOLDER = 'dataset'

model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, 
                                                        torch_dtype=torch.bfloat16, 
                                                        device_map="cuda",
                                                        attn_implementation="flash_attention_2")

model = torch.compile(model, mode="max-autotune")
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

def parse_receipt(image_path, entity_name, entity_units):
    prompt = f"""You are an AI assistant specialized in analyzing images. Your task is to extract specific information from the given image.
Please follow these instructions:
1. Look for the entity named "{entity_name}" on the image.
2. If found, extract the value associated with this entity.
3. One of these units may follow the value: {entity_units}.
4. Return only the numerical value and associated unit, if applicable.
5. If the entity is not found respond with "Not found".
Examples:
- If asked for "item_weight" and you find "Net Wt: 500g", return "500 gram"
- If asked for "item_volume" and you find "1 Cup a day", return "1 cup"
Remember:
- Be precise and return only the requested information.
- Do not include any additional text or explanations in your response.
    """
    messages = [
    {
        "role": "user",
        "content": [            
            {"type": "text", "text": prompt},
            {"type": "image", "image": image_path}
        ],
    }
]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text],images=image_inputs,videos=video_inputs,padding=True,return_tensors="pt").to("cuda")
    generation_args = {
        "max_new_tokens": 20,
        "temperature": 0.0,
        "do_sample": False,
    }
    generate_ids = model.generate(**inputs, **generation_args)
    generate_ids = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generate_ids)]
    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return response

def predictor(args):
    image_link, entity_name = args
    return parse_receipt(f"test/{image_link.split('/')[-1]}", entity_name, entity_unit_map[entity_name])

if __name__ == '__main__':
    test = pd.read_csv(os.path.join(DATASET_FOLDER, 'test.csv')).head(64)
    test['prediction'] = test.progress_apply(lambda row: predictor((row['image_link'], row['entity_name'])), axis=1)    
    output_filename = os.path.join(DATASET_FOLDER, 'test_out_qwen2b.csv')
    test[['index', 'prediction']].to_csv(output_filename, index=False)
