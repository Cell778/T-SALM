from pathlib import Path
import ollama
import json
from ollama import Client
import multiprocessing as mp
import random
import tqdm

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def caprtion_gen(metafiles, port):

    log = open(f'datasets/spatial_audio_text/{dataset_name}/{metadata_dir}/caption_generation_{port}.txt', 'w')

    client = Client(
        host=f'http://localhost:{port}',
        )
    its = tqdm.tqdm(enumerate(metafiles), total=len(metafiles), unit='clips', desc=f'PORT: {port}')
    for idx, metafile in its:
        with open(metafile) as f:
            json_data = json.load(f)
        captions = json_data['caption']

        if -22.5 < json_data['azi'] <= 22.5: direction = 'south'
        elif 22.5 < json_data['azi'] <= 67.5: direction = 'southeast'
        elif 67.5 < json_data['azi'] <= 112.5: direction = 'east'
        elif 112.5 < json_data['azi'] <= 157.5: direction = 'northeast'
        elif -22.5 > json_data['azi'] >= -67.5: direction = 'southwest'
        elif -67.5 > json_data['azi'] >= -112.5: direction = 'west'
        elif -112.5 > json_data['azi'] >= -157.5: direction = 'northwest'
        else: direction = 'north'

        spatialized_caption = []
        for idy, caption in enumerate(captions):
            prompt = [
                {'role': 'system', 'content': suffix},
                {'role': 'user', 'content': f'The sound "{caption}" is coming from the {direction}.'}
            ]
            response: ollama.GenerateResponse = client.chat(
                model='llama3.3', 
                messages=prompt, 
                options={'temperature': 0.2}
            )
            content = response.message.content
            if direction not in content:
                log.write(f'{str(metafile)}, {idy}\n')
            spatialized_caption.append(content)

        json_data['spatialized_caption'] = spatialized_caption
        json_data['direction'] = 'The sound is coming from the ' + direction + '.'
        with open(metafile, 'w') as f:
            json.dump(json_data, f, indent=4)
        print('PORT:', port, f'{idx+1}/{len(metafiles)}\n', metafile)

    log.close()

if __name__ == '__main__':
    dataset_name = 'AudioCaps'
    metadata_dir = 'metadata_temp0.2_llama3.3'
    num_workers = 1
    PORTS = [12355]
    # PORTS = [12345, 12346, 12348, 12349]
    # PORTS = [12355, 12356, 12357, 12358]
    assert num_workers == len(PORTS), 'Number of workers and ports should be the same.'


    metafiles = list(Path(f'/home/hjb/workspace/Spatial-CLAP/datasets/spatial_audio_text/{dataset_name}/{metadata_dir}').rglob('*.json'))
    random.shuffle(metafiles)

    suffix = 'Rephrase the sentence in English to concisely describe the sound detail and the direction of its source.'

    pool = [mp.Process(target=caprtion_gen, args=(metafiles[i::num_workers], PORTS[i])) for i in range(num_workers)]
    for p in pool: p.start()
    for p in pool: p.join()

