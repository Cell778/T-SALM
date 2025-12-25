# python datasim/audio_simulation.py --dataset=Freesound --dataset_type=train --num_workers=64 --chunksize=64
# python datasim/audio_simulation.py --dataset=Freesound --dataset_type=test --num_workers=64 --chunksize=64

python datasim/audio_simulation.py --dataset=AudioCaps --dataset_type=train 
python datasim/audio_simulation.py --dataset=AudioCaps --dataset_type=valid 
python datasim/audio_simulation.py --dataset=AudioCaps --dataset_type=test 

# python datasim/audio_simulation.py --dataset=Clotho --dataset_type=train --num_workers=1 --chunksize=1
# python datasim/audio_simulation.py --dataset=Clotho --dataset_type=valid --num_workers=1 --chunksize=1
# python datasim/audio_simulation.py --dataset=Clotho --dataset_type=test --num_workers=1 --chunksize=1
