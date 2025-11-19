import torch
import torch.nn as nn
import torchaudio

eps = torch.finfo(torch.float32).eps
window_fn_dict = {
    'hann': torch.hann_window,
    'hamming': torch.hamming_window,
    'blackman': torch.blackman_window,
    'bartlett': torch.bartlett_window,
}

class LogmelIV_Extractor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        data = cfg['data']
        assert data['window'] in window_fn_dict.keys(), \
            "window must be in {}, but got {}".format(window_fn_dict.keys(), data['window'])
        
        self.stft_extractor = torchaudio.transforms.Spectrogram(
            n_fft=data['nfft'], hop_length=data['hoplen'], 
            win_length=data['nfft'], window_fn=window_fn_dict[data['window']], 
            power=None,)
        self.mel_scale = torchaudio.transforms.MelScale(
            n_mels=data['n_mels'], sample_rate=data['sample_rate'], norm='slaney',
            f_min=20, f_max=data['sample_rate']/2, n_stft=data['nfft']//2+1,)
        self.amp2db = torchaudio.transforms.AmplitudeToDB(
            stype='power', top_db=None, )
        self.intensity_vector = intensityvector
    
    def forward(self, x):
        """
        input: 
            (batch_size, channels=4, data_length)
        output: 
            (batch_size, channels, time_steps, freq_bins) freq_bins->mel_bins
        """
        if x.ndim != 3:
            raise ValueError("x shape must be (batch_size, num_channels, data_length)\n \
                            Now it is {}".format(x.shape))
        x = self.stft_extractor(x)
        mel = self.mel_scale(torch.abs(x)**2)
        logmel = self.amp2db(mel).transpose(-1, -2)
        intensity_vector = self.intensity_vector(
            [x.real.transpose(-1, -2), x.imag.transpose(-1, -2)], 
            self.mel_scale.fb)
        out = torch.cat((logmel, intensity_vector), dim=1)
        return out


class Logmel_Extractor(nn.Module):
    
    def __init__(self, cfg):
        super().__init__()
        
        data = cfg['data']
        assert data['window'] in window_fn_dict.keys(), \
            "window must be in {}, but got {}".format(window_fn_dict.keys(), data['window'])
        
        self.stft_extractor = torchaudio.transforms.Spectrogram(
            n_fft=data['nfft'], hop_length=data['hoplen'], 
            win_length=data['nfft'], window_fn=window_fn_dict[data['window']], 
            power=None,)
        self.mel_scale = torchaudio.transforms.MelScale(
            n_mels=data['n_mels'], sample_rate=data['sample_rate'], norm='slaney',
            f_min=20, f_max=data['sample_rate']/2, n_stft=data['nfft']//2+1,)
        self.amp2db = torchaudio.transforms.AmplitudeToDB(
            stype='power', top_db=None, )
    
    def forward(self, x):
        """
        input: 
            (batch_size, channels=4, data_length)
        output: 
            (batch_size, channels, time_steps, freq_bins) freq_bins->mel_bins
        """
        if x.ndim != 3:
            raise ValueError("x shape must be (batch_size, num_channels, data_length)\n \
                            Now it is {}".format(x.shape))
        x = self.stft_extractor(x)
        mel = self.mel_scale(torch.abs(x)**2)
        logmel = self.amp2db(mel).transpose(-1, -2)
        out = logmel
        return out


def intensityvector(input, melW):
    """Calculate intensity vector. Input is four channel stft of the signals.
    input: (stft_real, stft_imag)
        stft_real: (batch_size, 4, time_steps, freq_bins)
        stft_imag: (batch_size, 4, time_steps, freq_bins)
    out:
        intenVec: (batch_size, 3, time_steps, freq_bins)
    """
    sig_real, sig_imag = input[0], input[1]
    Pref_real, Pref_imag = sig_real[:,0,...], sig_imag[:,0,...]
    Px_real, Px_imag = sig_real[:,1,...], sig_imag[:,1,...]
    Py_real, Py_imag = sig_real[:,2,...], sig_imag[:,2,...]
    Pz_real, Pz_imag = sig_real[:,3,...], sig_imag[:,3,...]

    IVx = Pref_real * Px_real + Pref_imag * Px_imag
    IVy = Pref_real * Py_real + Pref_imag * Py_imag
    IVz = Pref_real * Pz_real + Pref_imag * Pz_imag
    normal = torch.sqrt(IVx**2 + IVy**2 + IVz**2) + eps

    IVx_mel = torch.matmul(IVx / normal, melW)
    IVy_mel = torch.matmul(IVy / normal, melW)
    IVz_mel = torch.matmul(IVz / normal, melW)
    intenVec = torch.stack([IVx_mel, IVy_mel, IVz_mel], dim=1)

    return intenVec
