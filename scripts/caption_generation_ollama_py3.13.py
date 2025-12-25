import argparse
from pathlib import Path
import ollama
import json
from ollama import Client
import multiprocessing as mp
import random
import tqdm
import subprocess

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
CORRECT = True

def caprtion_gen(metafiles, port, dataset_name, metadata_dir, suffix):
    # 确保日志目录存在，然后打开日志文件
    log_dir = Path(f'datasets/spatial_audio_text/{dataset_name}/{metadata_dir}')
    log_dir.mkdir(parents=True, exist_ok=True)
    log = open(log_dir / f'caption_generation_{port}.txt', 'w', encoding='utf-8')

    client = Client(host=f'http://127.0.0.1:{port}')
    # 子进程不要再 pull（主进程已 pull）
    its = tqdm.tqdm(enumerate(metafiles), total=len(metafiles), unit='clips', desc=f'PORT: {port}')
    for idx, metafile in its:
        try:
            with open(metafile, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
        except Exception as e:
            log.write(f'LOAD_ERROR {metafile}: {e}\n')
            continue

        captions = json_data.get('caption', [])
        azi = json_data.get('azi', 0)
        if -22.5 < azi <= 22.5: direction = 'south'
        elif 22.5 < azi <= 67.5: direction = 'southeast'
        elif 67.5 < azi <= 112.5: direction = 'east'
        elif 112.5 < azi <= 157.5: direction = 'northeast'
        elif -22.5 > azi >= -67.5: direction = 'southwest'
        elif -67.5 > azi >= -112.5: direction = 'west'
        elif -112.5 > azi >= -157.5: direction = 'northwest'
        else: direction = 'north'

        if CORRECT and Path(str(metafile).replace('/metadata/', '/metadata_llama3.2/')).exists():
            print(f'[llama3.2] Caption already exists for {metafile}')
            continue

        spatialized_caption = []
        for idy, caption in enumerate(captions):
            prompt = [
                {'role': 'system', 'content': suffix},
                {'role': 'user', 'content': f'The sound "{caption}" is coming from the {direction}.'}
            ]
            try:
                response = client.chat(model='llama3.2', messages=prompt, options={'temperature': 0.2})
                content = response.message.content
                content = str(content).split("</think>")[-1].replace('\n','').strip()
            except Exception as e:
                content = ""
                log.write(f'GEN_ERROR {metafile}, {idy}: {e}\n')

            if direction not in content:
                log.write(f'{str(metafile)}, {idy}\n')
            spatialized_caption.append(content)

        json_data['spatialized_caption'] = spatialized_caption
        json_data['direction'] = 'The sound is coming from the ' + direction + '.'

        out_path = str(metafile).replace('/metadata/', '/metadata_llama3.2/')
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=4, ensure_ascii=False)

    log.close()

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='Clotho')
    parser.add_argument('--num_workers', type=int, default=1)
    # parser.add_argument('--ports', type=int, nargs='+', default=[11434, 12357])
    parser.add_argument('--ports', type=int, nargs='+', default=[11434])
    parser.add_argument('--dataset_type', type=str, default='none')
    args = parser.parse_args()

    dataset_name = args.dataset_name
    num_workers = args.num_workers
    PORTS = args.ports
    metadata_dir = 'metadata'
    if args.dataset_type != 'none':
        metadata_dir = f'metadata/{args.dataset_type}'
    assert num_workers == len(PORTS), 'Number of workers and ports should be the same.'

    metafiles = list(Path(f'datasets/spatial_audio_text/{dataset_name}/{metadata_dir}').rglob('*.json'))
    random.shuffle(metafiles)

    suffix = 'Rephrase the sentence in English to concisely describe the sound detail and the direction of its source.'

    # 主进程预先拉取模型一次（若已手动 pull，可注释）
    try:
        subprocess.run(["ollama", "pull", "llama3.2"], check=True)
    except Exception as e:
        print("ollama pull failed or already pulled:", e)

    # 把必要参数传入子进程，避免 spawn 导致名称不可见
    pool = []
    for i in range(num_workers):
        batch = metafiles[i::num_workers]
        p = mp.Process(target=caprtion_gen, args=(batch, PORTS[i], dataset_name, metadata_dir, suffix))
        pool.append(p)

    for p in pool: p.start()
    for p in pool: p.join()

