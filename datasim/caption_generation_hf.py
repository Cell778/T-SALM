from pathlib import Path
import json
from tqdm import tqdm
from transformers import pipeline
import torch
from tqdm import tqdm

model_id='meta-llama/Llama-3.2-3B-Instruct'
metafiles = list(Path('datasets/spatial_audio_text/AudioCaps/metadata').rglob('*.json'))
instruction = 'Rephrase the sentence as a short English sentence describing the sound and all the details of its source.'
BATCH_SIZE = 128

pipe = pipeline("text-generation", model=model_id, 
                batch_size=BATCH_SIZE, device=0,
                torch_dtype=torch.bfloat16)
pipe.tokenizer.pad_token = pipe.tokenizer.eos_token

metadata = []
for metafile in metafiles:
    with open(metafile) as f:
        json_data = json.load(f)
    if -22.5 < json_data['azi'] <= 22.5: direction = 'south'
    elif 22.5 < json_data['azi'] <= 67.5: direction = 'southeast'
    elif 67.5 < json_data['azi'] <= 112.5: direction = 'east'
    elif 112.5 < json_data['azi'] <= 157.5: direction = 'northeast'
    elif -22.5 > json_data['azi'] >= -67.5: direction = 'southwest'
    elif -67.5 > json_data['azi'] >= -112.5: direction = 'west'
    elif -112.5 > json_data['azi'] >= -157.5: direction = 'northwest'
    else: direction = 'north'
    json_data['direction'] = 'The sound is coming from the ' + direction + '.'
    captions = json_data['caption']
    messages = []
    for caption in captions:
        messages.append([
            {'role': 'system', 'content': instruction},
            {'role': 'user', 'content': f'The sound \"{caption}\" is coming from the {direction}.'}
        ])
    metadata.append((metafile, json_data, messages, len(messages)))

progress_bar = tqdm(total=len(metadata))
for i in range(0, len(metadata), BATCH_SIZE):
    batch = metadata[i:i+BATCH_SIZE]
    batch_prompts = []
    for data in batch:
        batch_prompts.extend(data[-2])
    spatialized_captions = pipe(batch_prompts, max_length=256, return_full_text=False)
    start = 0
    for _, (metafile, json_data, message, num_cap) in enumerate(batch):
        caps = spatialized_captions[start:start+num_cap]
        json_data['spatialized_caption'] = [cap[0]['generated_text'] for cap in caps]
        with open(metafile, 'w') as f:
            json.dump(json_data, f, indent=4)
        tqdm.write(f'Generated caption for {metafile}')
        start += num_cap
        progress_bar.update(1)


