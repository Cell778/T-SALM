import json
import sys
sys.path.append('/home/hjb/workspace/Spatial-CLAP/src')
sys.path.append('/home/hjb/workspace/Spatial-CLAP/datasim')

import argparse
import librosa
import soundfile as sf
from pathlib import Path
import numpy as np
from itertools import product
import scipy.signal as scysignal
from scipy import special as scyspecial
import pyroomacoustics as pra
from tqdm.contrib.concurrent import process_map

from utils.config import get_dataset
from utilities import *


reg_type_dict = {
    'tikhonov': 'tikhonov',
    'open': 'soft',
    'rigid': 'hard'
    }
matlab_dir = '/Users/cellren/Desktop/XJTLU/SALM/Spatial-ALM/datasim/dependencies'
material_database = '/Users/cellren/Desktop/XJTLU/SALM/Spatial-ALM/datasets/material_absorption'
tau_srir_db = '/Users/cellren/Desktop/datasets/TAU-SRIR_DB/TAU-SRIR_DB'

# tau_srir_db = '/home/hjb/workspace/Spatial-CLAP/TAU-SRIR_DB'

# room_paths = ['01_bomb_shelter', '02_gym', '03_pb132', '04_pc226', '05_sa203',
#               '06_sc203', '08_se203', '09_tb103', '10_tc352']
# room_paths = ['01_bomb_shelter', '02_gym', '03_pb132', '04_pc226', '10_tc352']
room_paths = ['05_sa203', '06_sc203', '08_se203', '09_tb103',]
room_dict = {'01_bomb_shelter': 0, '02_gym': 1, '03_pb132': 2,
            '04_pc226': 3, '05_sa203': 4, '06_sc203': 5,
            '08_se203': 6, '09_tb103': 7, '10_tc352': 8}


class SRIRGenerator:
    """Spatial Room Impulse Response (SRIR) Generation. 
        Only support FOA (First Order Ambisonics) for now.

    """
    
    def __init__(self, fs=24000, radius=0.042, array_type='open', n_points=2048):
        
        self.fs = fs
        self.radius = radius
        self.array_type = array_type
        self.n_points = n_points
        self.SH_order = 1   

        # Microphone positions [n_mic, (azimuth, elevation)] in degree
        self.mic_pos_sph = np.array([[45, 35], [-45, -35], 
                                     [135, -35], [-135, 35]])
        self.mic_pos_cart = sph2cart(self.mic_pos_sph[:, 0], 
                                     self.mic_pos_sph[:, 1], self.radius)
        
        azi_rad = np.deg2rad(self.mic_pos_sph[:, 0])
        colat_rad = np.deg2rad(90 - self.mic_pos_sph[:, 1])
        self.SH_matrix = sh_matrix(N=1, azi=azi_rad, colat=colat_rad)
        
        f = np.linspace(0, fs//2, self.n_points//2+1)
        k = 2 * np.pi * f / 343

        self.bn = get_sma_radial_filters(
            k=k, reg_type=reg_type_dict[array_type], 
            r=radius, matlab_dir=matlab_dir)
        
        self.rir = None
        self.c = 343
    

    def compute_srir(self, room_dim, src_pos, rt60=None, 
                     mic_pos_center=None, method='hybrid', **kwargs):
        """Compute an SRIR from a given room parameters
            using pyroomacoustics.

        Parameters
        ----------
        room_dim : (3,) array_like
            Room dimensions in meters
        src_pos : (num_src, 3) array_like
            Source positions in meters
        rt60 : float,
            Desired RT60 in seconds
        mic_pos_center : (3,) array_like, optional
            Position of center of microphone array, 
            The default None, which means the center of room.
        method : str, {'ism', 'hybrid'}, optional
            Method of rir generator, by default 'hybrid', which
            means image source method and ray tracing are used.
        kwargs : dict
            Additional arguments for pyroomacoustics.ShoeBox

        Returns
        -------
        array_like, shape (num_mic, num_src, length)
            Generated SRIR.
        """        

        if mic_pos_center is None:
            mic_pos = self.mic_pos_cart.T + np.c_[room_dim]/2
        else:
            mic_pos = self.mic_pos_cart.T + np.c_[mic_pos_center]

        if rt60 is not None:
            # We invert Sabine's formula to obtain the parameters for the ISM simulator
            e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)
            max_order = min(max_order, 50)
            if method == "ism":
                room = pra.ShoeBox(
                    p=room_dim, 
                    fs=self.fs, 
                    materials=pra.Material(e_absorption), 
                    max_order=max_order,)
            elif method == "hybrid":
                room = pra.ShoeBox(
                    p=room_dim,
                    fs=self.fs,
                    materials=pra.Material(e_absorption),
                    max_order=max_order,
                    ray_tracing=True,
                    air_absorption=True,)
        else:
            enable = method == 'hybrid'
            room = pra.ShoeBox(
            p=room_dim, fs=self.fs, 
            ray_tracing=enable, 
            air_absorption=enable, 
            **kwargs)
        
        for pos in src_pos:
            room.add_source(pos)
        room.add_microphone_array(mic_pos)
        try:
            room.compute_rir()
        except:
            print(rt60, room_dim, src_pos, mic_pos_center, method, kwargs)
            raise ValueError('RIR computation failed.')

        self.rir = room.rir
        self.c = room.c


    def simulate(self, src_pos_mic, src_signals):
        """Simulates the microphone signal at every microphone in the array
        
        """

        assert len(src_pos_mic) == len(src_signals), \
            "Number of source position and signals must be equal."
        assert self.rir is not None, "Room impulse response is not computed."

        def simulate_rigid_sph_array(src_pos, n_points, order=30):
            """Rigid spherical array response simulation.

            Parameters
            ----------
            src_pos : (num_src, 3) array_like
                Source positions in meters
            n_points : int
                Points of filter.
            order : int, optional
                Order of expension term, by default 30
                The expansion is limited to 30 terms which provide 
                    negligible modeling error up to 20 kHz.

            Returns
            -------
            h_mic: (num_src, num_mic, n_points) array_like
                Spherical array response.
            """

            src_pos = np.asarray(src_pos)
            c = 343

            # Compute the frequency-dependent part of the microphone responses
            f = np.linspace(0, self.fs//2, n_points//2+1)
            kr = 2 * np.pi * f / c * self.radius

            b_n = np.zeros((order+1, len(f)), dtype=np.complex128)
            for n in range(order+1):
                b_n[n] = mode_strength(n=n, kr=kr, sphere_type=self.array_type)
            temp = b_n
            temp[:, -1] = np.real(temp[:, -1])
            temp = np.concatenate((temp, temp[:,-2:0:-1].conj()), axis=1)
            b_nt = np.fft.fftshift(np.fft.ifft(temp, axis=1), axes=1).real

            # Compute angular-dependent part of the microphone responses
            # unit vectors of DOAs and microphones
            N_doa = len(src_pos)
            N_mic = len(self.mic_pos_cart)
            h_mic = np.zeros((n_points, N_mic, N_doa))
            H_mic = np.zeros((n_points//2+1, N_mic, N_doa), dtype=np.complex128)
            for i in range(N_doa):
                cosAngle = np.dot(
                    self.mic_pos_cart / self.radius, 
                    src_pos[i,:] / np.linalg.norm(src_pos[i,:]))
                P = np.zeros((order+1, N_mic))
                for n in range(order+1):
                    Pn = scyspecial.lpmv(0, n, cosAngle)
                    P[n, :] = (2*n+1) / (4 * np.pi) * Pn
                
                h_mic[:,:,i] = b_nt.T @ P
                H_mic[:,:,i] = b_n.T @ P

            return h_mic.transpose(2,1,0), H_mic.transpose(2,1,0)

        num_src = len(src_pos_mic)
        num_mic = len(self.rir)

        max_len_rir = np.array(
            [len(self.rir[i][j]) for i, j in product(range(num_mic), range(num_src))]
        ).max()
        f = lambda i: len(src_signals[i])
        max_sig_len = np.array([f(i) for i in range(num_src)]).max()
        num_points = int(max_len_rir) + int(max_sig_len) - 1
        if num_points % 2 == 1:
            num_points += 1
        # the array that will receive all the signals
        premix_signals = np.zeros((num_src, num_mic, num_points))
        # compute the signal at every microphone in the array
        for m in np.arange(num_mic):
            for s in np.arange(num_src):
                sig = src_signals[s]
                if sig is None:
                    continue
                h = self.rir[m][s]
                premix_signals[s, m, :len(sig) + len(h) - 1] += \
                    scysignal.fftconvolve(h, sig)

        if self.array_type == 'open':
            return np.sum(premix_signals, axis=0)
        elif self.array_type == 'rigid':
            h_mic, H_mic = simulate_rigid_sph_array(src_pos_mic, n_points=self.n_points)
            return scysignal.fftconvolve(premix_signals, h_mic, axes=-1).sum(axis=0)
        else:
            raise ValueError('Array type not supported.')
    

    def mic2foa(self, signal, norm='SN3D', SH_type='real'):
        """Encode a audio signal to first-order ambisonics format.

        Parameters
        ----------
        signal : (num_mic, num_samples), array_like
            The input audio signal.
        norm : str, {'N3D', 'SN3D'}, optional
            Normalization of the SH basis, by default 'SN3D'

        Returns
        -------
        Signal_hoa: (num_mic, num_samples), array_like
            The encoded signal.
        """        
        if SH_type not in ['real', 'complex']:
            raise ValueError('SH_type must be \'real\' or \'complex\'.')       
        
        def N3D_to_SN3D(F_nm, sh_axis=0):
            """Convert N3D (orthonormal) to SN3D (Schmidt semi-normalized) signals.

            Parameters
            ----------
            F_nm : ((N_sph+1)**2, S) numpy.ndarray
                Matrix of spherical harmonics coefficients of spherical function(S).
            sh_axis : int, optional
                SH axis. The default is 0.

            Returns
            -------
            F_nm : ((N_sph+1)**2, S) numpy.ndarray
                Matrix of spherical harmonics coefficients of spherical function(S).

            """
            assert(F_nm.ndim == 2)
            # Input SH order
            N = int(np.sqrt(F_nm.shape[sh_axis]) - 1)
            # 1/sqrt(2n+1) conversion factor
            n_norm = np.array([1/np.sqrt(2*n + 1) for n in range(N + 1)])
            # Broadcast
            n_norm = np.expand_dims(repeat_per_order(n_norm), axis=sh_axis-1)
            return n_norm * F_nm
        
        b_n, b_n_inv, b_n_inv_t = self.bn

        if SH_type == 'real':
            # The convention used here is also known as N3D-ACN (for SH_type='real').
            signal = self.SH_matrix.T @ signal
        elif SH_type == 'complex':
            raise NotImplementedError('Complex SH_type is not implemented yet.')
        b_n_inv_t = np.repeat(b_n_inv_t, 2*np.arange(self.SH_order+1)+1, axis=0)
        signal_hoa = scysignal.fftconvolve(b_n_inv_t, signal, axes=-1)

        if norm == 'SN3D':
            return N3D_to_SN3D(signal_hoa)
        elif norm == 'N3D':
            return signal_hoa
        else:
            raise ValueError('Normalization type {} not supported'.format(norm))



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Clotho')
    parser.add_argument('--dataset_type', type=str, default='train')
    parser.add_argument('--num_workers', type=int, default=48)
    parser.add_argument('--chunksize', type=int, default=24)
    args = parser.parse_args()

    # Load the dataset
    dataset_name = args.dataset # 'Clotho'
    dataset_type = args.dataset_type # 'train', 'valid', 'test'
    dataset = get_dataset(dataset_name).dataset
    audiofiles = dataset[dataset_type]['audio']
    metadata = dataset[dataset_type]['text']

    # audiofiles = list(Path('/home/hjb/workspace/Data-synthesis/dataset/sound_events/maleSpeech/fsd50k/Male_speech_and_man_speaking').glob('*.wav'))
    # metadata = {file.name: 'caption' for file in audiofiles}
    # dataset_name = 'FSD50K'

    # Parameters for the simulation
    nb_pos_per_audio = 3
    tgt_sr = 24000
    room_size_range = [[4., 20.], [4., 20.], [3., 10.]]
    mic_pos_range_percentage = [0.4, 0.6] # percentage of the room size
    src_pos_from_walls = 0.5
    src_pos_from_listener = 1
    tgt_dir = Path(f'datasets/spatial_audio_text/{dataset_name}_ColRIR_New/audio/{dataset_type}')
    tgt_meta_dir = Path(f'datasets/spatial_audio_text/{dataset_name}_ColRIR_New/metadata/{dataset_type}')
    tgt_dir.mkdir(parents=True, exist_ok=True)
    tgt_meta_dir.mkdir(parents=True, exist_ok=True)
    rnd_generator = np.random.default_rng(seed=2025)

    # Load the material absorption database for room simulation
    absortpion_table = pra.materials_data['absorption']
    ceilings, floors, walls = [], [], []
    ceilings += list(absortpion_table['Ceiling absorbers'].keys())
    floors += list(absortpion_table['Floor coverings'].keys()) + \
        ['concrete_floor', 'marble_floor'] * 8 + ['audience_floor', 'stage_floor'] * 3
    walls += list(absortpion_table['Wall absorbers'].keys()) + \
        ['hard_surface', 'brickwork', 'brick_wall_rough', 'limestone_wall']
    ceilings += get_materials_absorption_database(material_database, 'ceiling')
    floors += get_materials_absorption_database(material_database, 'floor')
    walls += get_materials_absorption_database(material_database, 'wall')

    # Load TAU-SRIR DB if synthesizing the test set
    # if dataset_type in []:
    if dataset_type in ['test', 'valid', 'train']:
        import mat73
        import scipy
        print('Loading rirdata.mat...')
        rirdata = scipy.io.loadmat(tau_srir_db + '/rirdata.mat')['rirdata']['room'][0][0]
        rirs = []
        for i in range(len(room_paths)):
            rir_path = tau_srir_db + '/rirs_' + room_paths[i] + '.mat'
            print(f'{i}: Loading {room_paths[i]} RIRs.')
            rirs.append(mat73.loadmat(rir_path)['rirs']['foa'])

    srir_generator = SRIRGenerator(fs=tgt_sr, radius=0.042, array_type='rigid')
    # for i, audiofile in enumerate(audiofiles):
    def generate_spatial_audio(idx):
        
        # Load the audio signal
        audiofile = audiofiles[idx]
        fname = audiofile.stem
        audio, _ = librosa.load(audiofile, sr=tgt_sr)

        ceiling = rnd_generator.choice(ceilings)
        floor = rnd_generator.choice(floors)
        wall = rnd_generator.choice(walls, size=4)
        materials = pra.make_materials(ceiling=ceiling, floor=floor, east=wall[0], 
                                       west=wall[1], north=wall[2], south=wall[3])
        mic_pos_percentage = rnd_generator.uniform(*mic_pos_range_percentage)
        room_size = np.array([rnd_generator.uniform(*ranges) for ranges in room_size_range])
        
        # while True: # ensure the RT60 is valid
        #     rt60 = rnd_generator.uniform(0.1, 0.5) # in seconds
        #     try: pra.inverse_sabine(rt60, room_size)
        #     except:
        #         room_size = np.array([rnd_generator.uniform(*ranges) 
        #                               for ranges in room_size_range])
        #     else: break
        
        mic_pos = room_size * mic_pos_percentage
        
        # randomly select 5 source positions from 8 directions
        directions = list(range(0, 360, 45))
        directions = rnd_generator.choice(directions, size=nb_pos_per_audio, replace=False)

        if dataset_type in []:
        # if dataset_type in ['train', 'valid', 'test']:
            for i in range(nb_pos_per_audio): # 3 different source positions
            
                # azi = directions[i] - 180 + rnd_generator.uniform(-10, 10)
                azi = directions[i] - 180 + rnd_generator.uniform(-22.5, 22.5)
                if azi < -180: azi += 360
                # ele = rnd_generator.uniform(-25, 25)
                ele = rnd_generator.uniform(-45, 45)
                while True:
                    r = rnd_generator.uniform(src_pos_from_listener, max(room_size)/2)
                    x, y, z = sph2cart(azi, ele, r).squeeze() + mic_pos
                    if np.all(np.array([x, y, z]) < (room_size - src_pos_from_walls)) and \
                        np.all(np.array([x, y, z]) > src_pos_from_walls):
                        break
                src_pos = np.array([x, y, z])

                srir_generator.compute_srir(
                    room_dim=room_size, src_pos=src_pos[None], mic_pos_center=mic_pos,
                    materials=materials, max_order=100)
                # srir_generator.compute_srir(
                #     room_dim=room_size, src_pos=src_pos[None], mic_pos_center=mic_pos,
                #     rt60=rt60)
                signal = srir_generator.simulate((src_pos-mic_pos)[None], audio[None])
                signal = srir_generator.mic2foa(signal)[:, :len(audio)]
                sf.write(tgt_dir / f'{fname}_{i}.flac', signal.T, tgt_sr)

                _rt60 = pra.experimental.measure_rt60(
                    srir_generator.rir[0][0], fs=tgt_sr, decay_db=60)
                _rt20 = pra.experimental.measure_rt60(
                    srir_generator.rir[0][0], fs=tgt_sr, decay_db=20)
                _rt30 = pra.experimental.measure_rt60(
                    srir_generator.rir[0][0], fs=tgt_sr, decay_db=30)
                captions = metadata[audiofile.name]
                # caption = captions[i % len(captions)] # repeat the metadata if less than 5
                meta_json = {'ori_audiofile': audiofile.name, 'caption': captions,
                            'rt60': _rt60, 'rt20': _rt20, 'rt30': _rt30, #'tgt_rt60': rt60,
                            'azi': azi, 'ele': ele, 'r': r}
                with open(tgt_meta_dir / f'{fname}_{i}.json', 'w') as f:
                    json.dump(meta_json, f, indent=4)

        # elif dataset_type in ['test']:
        else:
            room_idx = rnd_generator.integers(0, len(room_paths))
            room_idy = room_dict[room_paths[room_idx]]
            n_traj, n_heights = rirdata[room_idy][0][3].shape
            traj_idx = rnd_generator.integers(0, n_traj)
            height_idx = rnd_generator.integers(0, n_heights)
            n_doas = rirdata[room_idy][0][3][traj_idx][height_idx].item()
            num_per_split = n_doas // nb_pos_per_audio
            for i in range(nb_pos_per_audio):
                doa_idx = rnd_generator.integers(
                    i*num_per_split, (i+1)*num_per_split)
                x, y, z = rirdata[room_idy][0][2][traj_idx][height_idx][0][doa_idx]
                azi, ele, _ = cart2sph(x, y, z).squeeze()
                rir = rirs[room_idx][traj_idx][height_idx][..., doa_idx]
                # signal = np.zeros((rir.shape[1], len(audio)))
                # for n_ch in range(rir.shape[1]):
                #     signal[n_ch, :] = scysignal.fftconvolve(rir[..., n_ch], audio)[:len(audio)]
                signal = scysignal.fftconvolve(rir.T, audio[None], axes=-1)
                sf.write(tgt_dir / f'{fname}_{i}.flac', signal.T, tgt_sr)
                captions = metadata[audiofile.name]
                meta_json = {'ori_audiofile': audiofile.name, 'caption': captions,
                            'azi': int(azi), 'ele': int(ele), 'room_idx': int(room_idx), 'n_traj': int(traj_idx),
                            'height_idx': int(height_idx), 'doa_idx': int(doa_idx)}
                with open(tgt_meta_dir / f'{fname}_{i}.json', 'w') as f:
                    json.dump(meta_json, f, indent=4)
                
        print(f'{idx}: {fname}.wav')

    # process_map(generate_spatial_audio, range(len(audiofiles)), 
    #             max_workers=args.num_workers, 
    #             chunksize=args.chunksize)
    
    from tqdm import tqdm
    for idx in tqdm(range(len(audiofiles)), desc=f"Processing {dataset_name} {dataset_type}"):
        generate_spatial_audio(idx)



