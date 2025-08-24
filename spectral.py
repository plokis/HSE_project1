import numpy as np

def compute_acf_numpy(signal1, signal2):
    n = len(signal1)
    acf = np.zeros(n)
    for tau in range(1, n + 1):
        acf[tau - 1] = np.sum(signal1[:n - tau + 1] * signal2[tau - 1:n]) / n
    return acf

def ampl_fft(signal):
    """
    Performs spectrum analysis of a signal.

    Parameters:
    - signal: 1D numpy array of complex signal values

    Returns:
    - spectr: array of spectrum values (complex)
    - freq: corresponding frequencies (Hz)
    """

    N = signal.shape[0]
    T = 0.00273

    m = signal.copy()

    specter = np.fft.fftn(m)

    # Расчёт частот вручную, как в оригинальной функции
    omega = np.zeros(N)
    t = np.zeros(N)

    for k in range(N):
        if k == 0:
            t[k] = 0
            omega[k] = 0
        elif k <= N // 2:
            t[k] = N * T / (k)
            omega[k] = 2 * np.pi / t[k]
        else:
            t[k] = N * T / (N - k)
            omega[k] = -2 * np.pi / t[k]

    # Преобразуем в частоты в Гц (omega / 2π)
    freq = omega / (2 * np.pi)

    # Сдвиг аналогичный MATLAB circshift
    shift = [1, N // 2 - 1]  # [2, floor(N/2)] в MATLAB (индексация с 1)
    freq = np.roll(freq, shift[1])
    freq = np.roll(freq, shift[0])
    specter = np.roll(specter, shift[1])
    specter = np.roll(specter, shift[0]) / N

    return specter, freq
