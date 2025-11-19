import torch
import torchvision
from torch.utils.data import Dataset
from transformers import RobertaTokenizer
import torch.nn.functional as F

class BaseDataset(Dataset):
    """ Base dataset for spatial audio understanding task

    """
    def __init__(self, cfg, dataset_name, splits=None, dataset_type='train'):
        """
        Args:
            cfg: configurations
            dataset_name: dataset used
            splits: splits to be used.
            dataset_type: 'train' | 'valid' | 'test' .
                'train' and 'val' (optional) are only used while training. 
                Either 'val' or 'test' is only used while infering.
        """
        super().__init__()

        self.cfg = cfg
        self.dataset_name = dataset_name
        self.dataset_type = dataset_type
        self.sample_rate = cfg.data.sample_rate
        self.chunklen_sec = 10
        self.chunklen = self.sample_rate * self.chunklen_sec
        self.shrink_mono = torchvision.transforms.Resize([1, self.chunklen])
        self.shrink_poly = torchvision.transforms.Resize([4, self.chunklen])
        self.audiofiles = []
        
        # Get tokenizer
        self.text_embed = None # for zero-shot classification
        if cfg.model.text.backbone == 'roberta':
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        else: raise NotImplementedError

    def tokenize(self, text, max_length=77):
        """tokenizer for different models
        tmodel is default to roberta as it is the best model for our task
        max_length is default to 77 from the OpenAI CLIP parameters
        We assume text to be a single string, but it can also be a list of strings
        """
        if self.cfg.model.text.backbone == 'roberta':
            result = self.tokenizer(text, max_length=max_length, return_tensors='pt',
                                    padding='max_length', truncation=True)
        else: raise NotImplementedError

        return {k: v.squeeze(0) for k, v in result.items()}

    def __len__(self):
        """Get length of the dataset

        """
        return len(self.audiofiles)

    def __getitem__(self, idx):
        """
        Read features from the dataset
        """
        raise NotImplementedError

    def data_filling(self, audio):
        """
        Fill the data in the dataset
        """
        if self.cfg.data.data_filling == 'repeatpad':
            n_repeats = int(self.chunklen / audio.shape[-1])
            audio = audio.repeat(1, n_repeats)
            audio = F.pad(audio, (0, self.chunklen - audio.shape[-1]), 
                          mode='constant', value=0)
        elif self.cfg.data.data_filling == 'pad':
            audio = F.pad(audio, (0, self.chunklen - audio.shape[-1]), 
                          mode='constant', value=0)
        elif self.cfg.data.data_filling == 'repeat':
            n_repeats = int(self.chunklen / audio.shape[-1])
            audio = audio.repeat(1, n_repeats + 1)[:, :self.chunklen]
        else: 
            raise ValueError('Unknown data filling method: {}'\
                             .format(self.cfg.data.data_filling))
        return audio
    
    def data_truncation_fusion(self, audio):
        """
        Truncate the data in the dataset
        """
        audio_shrink = self.shrink_mono(audio[None]).squeeze(0)
        ranges = torch.tensor_split(torch.arange(audio.shape[-1] - self.chunklen), 3)
        ranges = [r.tolist() for r in ranges]
        if len(ranges[0]) < 2: 
            # if the audio is too short, we just use the first chunk
            ranges[0] = [0, 1]
        if len(ranges[1]) < 2: 
                # if the audio is too short, we just use the first chunk
            ranges[1] = [0, 1]
        if len(ranges[2]) < 2: 
                # if the audio is too short, we just use the first chunk
            ranges[2] = [0, 1]
        # randomly choose index for each part
        # NOTE: increase the randomness while using multiple num_workers
        idx_front = torch.randint(low=ranges[0][0], high=ranges[0][-1], size=(1,)).item()
        idx_middle = torch.randint(low=ranges[1][0], high=ranges[1][-1], size=(1,)).item()
        idx_end = torch.randint(low=ranges[2][0], high=ranges[2][-1], size=(1,)).item()
        
        # select the chunk
        audio_front = audio[:, idx_front:idx_front+self.chunklen]
        audio_middle = audio[:, idx_middle:idx_middle+self.chunklen]
        audio_end = audio[:, idx_end:idx_end+self.chunklen]

        audio = torch.cat([audio_shrink, audio_front, audio_middle, audio_end], dim=0)
        return audio

