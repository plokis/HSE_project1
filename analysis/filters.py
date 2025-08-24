import numpy as np

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

def manual_convolution(array, kernel):
    array_len = len(array)
    kernel_len = len(kernel)
    pad_width = kernel_len // 2
    padded_array = np.pad(array, pad_width, mode='edge')
    result = np.zeros_like(array)
    for j in range(array_len):
        result[j] = np.sum(padded_array[j:j + kernel_len] * kernel)
    return result

def pantelleev_filter(t, omega_0: float):
    t = np.asarray(t)
    exp_term = np.exp(-omega_0 * np.abs(t) / np.sqrt(2))
    cos_term = (3 / np.sqrt(2)) * np.cos(omega_0 * t / np.sqrt(2))
    sin_term = (omega_0 * np.abs(t) + 3 / np.sqrt(2)) * np.sin(omega_0 * np.abs(t) / np.sqrt(2))
    h_t = (omega_0 / 8) * exp_term * (cos_term + sin_term)
    return h_t