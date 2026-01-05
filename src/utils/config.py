from utils.datasets import *
import utils.feature as feature


dataset_dict = {
    'Clotho': Clotho,
    'AudioCaps': AudioCaps,
    'ESC50': ESC50,
    'Freesound': Freesound,
    'sClotho': sClotho,
    'stClotho': stClotho,
    'sClotho_ColRIR': sClotho,
    'sClotho_ColRIR_New': sClotho,
    'sAudioCaps': sAudioCaps,
    'sAudioCaps_ColRIR': sAudioCaps,
    'sAudioCaps_ColRIR_New': sAudioCaps,
    'sFreesound': sFreesound,
}

# Datasets
def get_dataset(dataset_name, cfg=None):
    dataset = dataset_dict[dataset_name](cfg, dataset_name=dataset_name)
    print('\nDataset {} is being developed......\n'.format(dataset_name))
    return dataset


def get_afextractor(cfg, audio_feature='logmel'):
    """ Get audio feature extractor."""
    if audio_feature == 'logmelIV':
        afextractor = feature.LogmelIV_Extractor(cfg)
    elif audio_feature == 'logmel':
        afextractor = feature.Logmel_Extractor(cfg)
    else:
        raise ValueError('Unknown audio feature extractor: {}'\
                         .format(cfg['data']['audio_feature']))
    return afextractor