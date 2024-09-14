import os
import pandas as pd
import torch
import warnings
from tqdm.auto import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from constants import entity_unit_map
from accelerate import PartialState
from accelerate.utils import gather_object

warnings.filterwarnings("ignore")
tqdm.pandas()

model_id = "Qwen/Qwen2-VL-7B-Instruct"
DATASET_FOLDER = 'dataset'

# Start up the distributed environment without needing the Accelerator
distributed_state = PartialState()

model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id, 
    device_map=distributed_state.device,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2"
).eval()

model = torch.compile(model, mode="max-autotune")
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

def parse_receipt(batch, model, processor):
    results = []
    for image_path, entity_name, index in batch:
        entity_units = entity_unit_map[entity_name]
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
                    {"type": "image", "image": f"test/{image_path.split('/')[-1]}", "resized_height": 768, "resized_width": 1280}
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(model.device)
        
        generation_args = {
            "max_new_tokens": 20,
            "temperature": 0.0,
            "do_sample": False,
        }
        generate_ids = model.generate(**inputs, **generation_args)
        generate_ids = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generate_ids)]
        response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        results.append((index, response))
    
    return results

def main():
    test = pd.read_csv(os.path.join(DATASET_FOLDER, 'test.csv'))
    
    # Prepare data in batches
    batch_size = 31  # Adjust based on your GPU memory
    data = list(zip(test['image_link'], test['entity_name'], test.index))
    batches = [data[i:i + batch_size + 1] for i in range(0, len(data), batch_size)]

    completions_per_process = []

    with distributed_state.split_between_processes(batches, apply_padding=False) as split_batches:
        for batch in tqdm(split_batches, disable=not distributed_state.is_local_main_process):
            batch_results = parse_receipt(batch, model, processor)
            completions_per_process.extend(batch_results)

    # Gather results from all processes
    completions_gather = gather_object(completions_per_process)

    if distributed_state.is_main_process:
        result_df = pd.DataFrame(completions, columns=['index', 'prediction'])        
        result_df = result_df.set_index('index').sort_index()
        result_df['index']+=1
        
        # Save the results
        output_filename = os.path.join(DATASET_FOLDER, 'test_out_qwen_7b_distributed.csv')
        result_df.to_csv(output_filename)

    distributed_state.print("Inference completed and results saved.")

if __name__ == '__main__':
    main()
