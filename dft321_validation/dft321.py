import numpy as np


def dft321(u):
    """
    Python reproduction of official DFT321f.m.

    Parameters
    ----------
    u : ndarray, shape [N, 3]
        Three-axis acceleration signal.

    Returns
    -------
    y : ndarray, shape [N]
        DFT321 combined one-dimensional signal.
    """
    u = np.asarray(u)

    if u.ndim != 2 or u.shape[1] != 3:
        raise ValueError(
            f"Expected input shape [N, 3], got {u.shape}"
        )

    n = u.shape[0]

    # MATLAB: U_temp = fft(u)
    U = np.fft.fft(u, axis=0)

    odd = n % 2

    if odd:
        n_half = (n + 1) // 2
    else:
        n_half = n // 2 + 1

    U_pos = U[:n_half, :]

    # MATLAB:
    # absY = real(sqrt(sum(U_temp .* conj(U_temp), 2)))
    magnitude = np.real(
        np.sqrt(
            np.sum(U_pos * np.conj(U_pos), axis=1)
        )
    )

    # MATLAB:
    # PhaseChoice = angle(sum(U_temp, 2))
    phase = np.angle(
        np.sum(U_pos, axis=1)
    )

    Y_pos = magnitude * np.exp(1j * phase)

    # Reconstruct conjugate-symmetric spectrum
    if odd:
        Y = np.concatenate([
            Y_pos,
            np.conj(Y_pos[-1:0:-1])
        ])
    else:
        Y = np.concatenate([
            Y_pos,
            np.conj(Y_pos[-2:0:-1])
        ])

    # MATLAB: y = real(ifft(Y))
    y = np.real(np.fft.ifft(Y))

    return y
