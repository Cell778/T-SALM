import numpy as np 
from scipy import special as scyspecial
import os
import pandas as pd

def sph2cart(azimuth, elevation, r, type='degree'):
    r"""
    Convert spherical to cartesian coordinates
    """
    assert type in ['degree', 'radian'], "Type must be 'degree' or 'radian'"
    if type == 'degree':
        azimuth = azimuth / 180.0 * np.pi
        elevation = elevation / 180.0 * np.pi

    x = r * np.cos(elevation) * np.cos(azimuth)
    y = r * np.cos(elevation) * np.sin(azimuth)
    z = r * np.sin(elevation)

    return np.c_[x, y, z]


def cart2sph(x, y, z, type='degree'):
    r"""
    Convert cartesian to spherical coordinates    
    """
    assert type in ['degree', 'radian'], "Type must be 'degree' or 'radian'"

    azimuth = np.arctan2(y,x)
    elevation = np.arctan2(z,np.sqrt(x**2 + y**2))
    r = np.sqrt(x**2 + y**2 + z**2)

    if type == 'degree':
        azimuth = azimuth / np.pi * 180.0
        elevation = elevation / np.pi * 180.0

    return np.c_[azimuth, elevation, r]


def asarray_1d(a, **kwargs):
    r"""Squeeze the input and check if the result is one-dimensional.

    Returns *a* converted to a `numpy.ndarray` and stripped of
    all singleton dimensions.  Scalars are "upgraded" to 1D arrays.
    The result must have exactly one dimension.
    If not, an error is raised.
    """
    result = np.squeeze(np.asarray(a, **kwargs))
    if result.ndim == 0:
        result = result.reshape((1,))
    elif result.ndim > 1:
        raise ValueError("array must be one-dimensional")
    return result


def repeat_per_order(c):
    r"""Repeat each coefficient in 'c' m times per spherical order n.

    Parameters
    ----------
    c : (N,) array_like
        Coefficients up to SH order N.

    Returns
    -------
    c_reshaped : ((N+1)**2,) array like
        Reshaped input coefficients.
    """
    c = asarray_1d(c)
    N = len(c) - 1
    return np.repeat(c, 2*np.arange(N+1)+1)


def segment_mixtures(signal, fs, start, end, clip_length=5):
    r"""
    If the duration of the signal is less than 5 seconds, pad the signal with zeros at the beginning and
    end. Otherwise, return the first 5 seconds of the signal
    """
    
    duration = np.shape(signal)[0] / fs
    if duration < clip_length:
        pad_width_before = int(np.ceil(start * fs))
        # pad_width_after = max(0, int(np.ceil(fs*(clip_length-end))))
        pad_width_after = int(max(0, clip_length * fs - np.shape(signal)[0] - pad_width_before))
        pad_width = ((pad_width_before, pad_width_after),)
        signal =  np.pad(signal, pad_width)
    
    if len(signal) < clip_length * fs:
        print(signal.shape, start, end, duration, pad_width)
        raise ValueError('Length of audio is not equal to the mixture duration.')
    
    return signal[:int(clip_length*fs)]


def sh_matrix(N, azi, colat, SH_type='real', weights=None):
    r"""Matrix of spherical harmonics up to order N for given angles.

    Computes a matrix of spherical harmonics up to order :math:`N`
    for the given angles/grid.

    REF: https://github.com/chris-hld/spaudiopy

    .. math::

        \mathbf{Y} = \left[ \begin{array}{ccccc}
        Y_0^0(\theta[0], \phi[0]) & Y_1^{-1}(\theta[0], \phi[0]) &
        Y_1^0(\theta[0], \phi[0]) &
        \dots & Y_N^N(\theta[0], \phi[0])  \\
        Y_0^0(\theta[1], \phi[1]) & Y_1^{-1}(\theta[1], \phi[1]) &
        Y_1^0(\theta[1], \phi[1]) &
        \dots & Y_N^N(\theta[1], \phi[1])  \\
        \vdots & \vdots & \vdots & \vdots & \vdots \\
        Y_0^0(\theta[Q-1], \phi[Q-1]) & Y_1^{-1}(\theta[Q-1], \phi[Q-1]) &
        Y_1^0(\theta[Q-1], \phi[Q-1]) &
        \dots & Y_N^N(\theta[Q-1], \phi[Q-1])
        \end{array} \right]

    where

    .. math::

        Y_n^m(\theta, \phi) = \sqrt{\frac{2n + 1}{4 \pi}
                                    \frac{(n-m)!}{(n+m)!}} P_n^m(\cos \theta)
                            e^{i m \phi}

    When using `SH_type='real'`, the real spherical harmonics
    :math:`Y_{n,m}(\theta, \phi)` are implemented as a relation to
    :math:`Y_n^m(\theta, \phi)`.

    Parameters
    ----------
    N : int
        Maximum SH order.
    azi : (Q,) array_like
        Azimuth.
    colat : (Q,) array_like
        Colatitude.
    SH_type :  'complex' or 'real' spherical harmonics.
    weights : (Q,) array_like, optional
        Quadrature weights.

    Returns
    -------
    Ymn : (Q, (N+1)**2) numpy.ndarray
        Matrix of spherical harmonics.

    Notes
    -----
    The convention used here is also known as N3D-ACN.

    """
    azi = asarray_1d(azi)
    colat = asarray_1d(colat)
    if azi.ndim == 0:
        Q = 1
    else:
        Q = len(azi)
    if weights is None:
        weights = np.ones(Q)
    if SH_type == 'complex':
        Ymn = np.zeros([Q, (N+1)**2], dtype=np.complex_)
    elif SH_type == 'real':
        Ymn = np.zeros([Q, (N+1)**2], dtype=np.float64)
    else:
        raise ValueError('SH_type unknown.')

    idx = 0
    for n in range(N+1):
        for m in range(-n, n+1):
            if SH_type == 'complex':
                Ymn[:, idx] = weights * scyspecial.sph_harm(m, n, azi, colat)
            elif SH_type == 'real':
                if m == 0:
                    Ymn[:, idx] = weights * np.real(
                                scyspecial.sph_harm(0, n, azi, colat))
                if m < 0:
                    Ymn[:, idx] = weights * np.sqrt(2) * (-1) ** abs(m) * \
                                np.imag(
                                scyspecial.sph_harm(abs(m), n, azi, colat))
                if m > 0:
                    Ymn[:, idx] = weights * np.sqrt(2) * (-1) ** abs(m) * \
                                np.real(
                                scyspecial.sph_harm(abs(m), n, azi, colat))

            idx += 1
    return Ymn


def get_sma_radial_filters(k, reg_type='tikhonov', r=0.042,
                           matlab_dir='./dependencies'):
    # REF: https://github.com/AppliedAcousticsChalmers/ambisonic-encoding
    import matlab
    import matlab.engine

    eng = matlab.engine.start_matlab()
    eng.cd(matlab_dir)
    print('Calculating radial filters...')

    b_n, b_n_inv, b_n_inv_t = eng.get_sma_radial_filters(
        matlab.double(k[:,None].tolist()), 
        float(r), float(1), 20.0, reg_type, 2, 
        nargout=3)
    b_n = np.array(b_n).T
    b_n_inv = np.array(b_n_inv).T
    b_n_inv_t = np.array(b_n_inv_t).T
    return b_n, b_n_inv, b_n_inv_t


def mode_strength(n, kr, sphere_type='rigid'):
    """Mode strength b_n(kr) for an incident plane wave on sphere.

    Parameters
    ----------
    n : int
        Degree.
    kr : array_like
        kr vector, product of wavenumber k and radius r_0.
    sphere_type : 'rigid' or 'open'

    Returns
    -------
    b_n : array_like
        Mode strength b_n(kr).

    References
    ----------
    Rafaely, B. (2015). Fundamentals of Spherical Array Processing. Springer.
    eq. (4.4) and (4.5).
    """

    np.seterr(divide='ignore', invalid='ignore')

    def spherical_hn2(n, z, derivative=False):
        """Spherical Hankel function of the second kind.

        Parameters
        ----------
        n : int, array_like
            Order of the spherical Hankel function (n >= 0).
        z : complex or float, array_like
            Argument of the spherical Hankel function.
        derivative : bool, optional
            If True, the value of the derivative (rather than the function
            itself) is returned.

        Returns
        -------
        hn2 : array_like


        References
        ----------
        http://mathworld.wolfram.com/SphericalHankelFunctionoftheSecondKind.html
        """
        with np.errstate(invalid='ignore'):
            yi = 1j * scyspecial.spherical_yn(n, z, derivative)
        return scyspecial.spherical_jn(n, z, derivative) - yi

    kr = np.asarray(kr)
    if sphere_type == 'open':
        b_n = 4*np.pi*1j**n * scyspecial.spherical_jn(n, kr)
    elif sphere_type == 'rigid':
        b_n = 4*np.pi*1j**n * (scyspecial.spherical_jn(n, kr) -
                            (scyspecial.spherical_jn(n, kr, True) /
                            spherical_hn2(n, kr, True)) *
                            spherical_hn2(n, kr))
    else:
        raise ValueError('sphere_type Not implemented.')
    
    idx_kr0 = np.where(kr==0)[0]
    idx_nan = np.where(np.isnan(b_n))[0]
    b_n[idx_nan] = 0
    if n == 0:
        b_n[idx_kr0] = 4*np.pi
    else:
        b_n[idx_kr0] = 0

    return b_n


def get_materials_absorption_database(root_path, surface):
    """ Get materials absorption database.
    """
    assert surface in ['ceiling', 'floor', 'wall'], 'Unknown surface type.'
    files = [file for file in os.listdir(root_path) if surface in file]
    materials = []
    for file in files:
        df = pd.read_csv(os.path.join(root_path, file)).values
        for item in df:
            material = {'description': item[0], 'coeffs': item[1:],
                        'center_freqs': [125, 250, 500, 1000, 2000, 4000]}
            materials.append(material)
    return materials