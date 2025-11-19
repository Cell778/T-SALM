from tqdm import tqdm
from pathlib import Path
import librosa
import soundfile as sf
from tqdm.contrib.concurrent import process_map

Clotho = list(Path('datasets/audio_text/Clotho').rglob('*.wav'))
AudioCaps = list(Path('datasets/audio_text/AudioCaps').rglob('*.wav'))
sClotho_dir = Path('datasets/spatial_audio_text/Clotho/audio')
sAudioCaps_dir = Path('datasets/spatial_audio_text/AudioCaps/audio')
# sClotho = list(Path('datasets/spatial_audio_text/Clotho').rglob('*.flac'))
# sAudioCaps = list(Path('datasets/spatial_audio_text/AudioCaps').rglob('*.flac'))

# for audiofile in tqdm(Clotho, desc='Clotho'):
def trunc_audio1(audiofile):
    dur = sf.info(audiofile).duration
    if '/development/' in str(audiofile): split = 'train'
    elif '/evaluation/' in str(audiofile): split = 'test'
    else: split = 'valid'
    for i in range(3):
        ori_file = sClotho_dir / split / f'{audiofile.stem}_{i}.flac'
        tgt_file = str(ori_file).replace('/train/', '/train_trunc/')\
                                .replace('/test/', '/test_trunc/')\
                                .replace('/valid/', '/valid_trunc/')   
        y, sr = librosa.load(ori_file, sr=None, duration=dur, mono=False)
        # Path(tgt_file).parent.mkdir(parents=True, exist_ok=True)
        sf.write(tgt_file, y.T, sr)
        print(tgt_file, sf.info(tgt_file).duration, sf.info(ori_file).duration, sf.info(audiofile).duration)

# for audiofile in tqdm(Clotho, desc='AudioCaps'):
def trunc_audio2(audiofile):
    dur = sf.info(audiofile).duration
    if '/train/' in str(audiofile): split = 'train'
    elif '/test/' in str(audiofile): split = 'test'
    else: split = 'valid'
    for i in range(3):
        ori_file = sAudioCaps_dir / split / f'{audiofile.stem}_{i}.flac'
        try:
            y, sr = librosa.load(ori_file, sr=None, duration=dur, mono=False)
        except:
            print(ori_file, 'warning----------------------------')
            y, sr = librosa.load(ori_file, sr=None, mono=False)
            y = y[:, :int(dur*sr)]
        tgt_file = str(ori_file).replace('/train/', '/train_trunc/')\
                                .replace('/test/', '/test_trunc/')\
                                .replace('/valid/', '/valid_trunc/')
        # Path(tgt_file).parent.mkdir(parents=True, exist_ok=True)
        sf.write(tgt_file, y.T, sr)
        print(tgt_file, sf.info(tgt_file).duration, sf.info(ori_file).duration, sf.info(audiofile).duration)

# process_map(trunc_audio1, Clotho, max_workers=32, chunksize=512)
process_map(trunc_audio2, AudioCaps, max_workers=128, chunksize=1024)
