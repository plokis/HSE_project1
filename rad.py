import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
from scipy.signal import argrelextrema
from scipy.stats import kstest, ks_2samp, mannwhitneyu, wilcoxon
from seaborn import histplot, heatmap

def compute_acf_numpy(signal1, signal2):
    n = len(signal1)
    acf = np.zeros(n)
    for tau in range(1, n + 1):
        acf[tau - 1] = np.sum(signal1[:n - tau + 1] * signal2[tau - 1:n]) / n
    return acf

def convert_decimal_year(decimal_year):
    year = int(decimal_year)  # Целая часть — это год
    month = int(np.ceil((decimal_year - year + 0.01) * 12))  # Доля года * 12 → номер месяца
    month = max(1, min(month, 12))  # Гарантируем, что месяц в диапазоне 1–12
    return f"{year:04d}-{month:02d}"

def averaged_by_interval(time_series, interval):
    return np.mean(time_series.reshape(-1, interval), axis=1)

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

def manual_convolution(array, kernel):
    array_len = len(array)
    kernel_len = len(kernel)
    pad_width = kernel_len // 2
    padded_array = np.pad(array, pad_width, mode='edge')  # Паддинг с повторением краёв
    result = np.zeros_like(array)

    for j in range(array_len):
        result[j] = np.sum(padded_array[j:j + kernel_len] * kernel)

    return result

def pantelleev_filter(t, omega_0: float):
    """
    Вычисляет значение функции h(t) по формуле Pantelleev filter.

    :param t: Число или массив значений времени t.
    :param omega_0: Параметр частоты (по умолчанию 1).
    :return: Значение h(t).
    """
    t = np.asarray(t)  # Преобразуем t в массив, если он не является массивом
    exp_term = np.exp(-omega_0 * np.abs(t) / np.sqrt(2))
    cos_term = (3 / np.sqrt(2)) * np.cos(omega_0 * t / np.sqrt(2))
    sin_term = (omega_0 * np.abs(t) + 3 / np.sqrt(2)) * np.sin(omega_0 * np.abs(t) / np.sqrt(2))

    h_t = (omega_0 / 8) * exp_term * (cos_term + sin_term)
    return h_t

def fit_sin(tt, yy):
    tt = np.array(tt)
    yy = np.array(yy)
    ff = np.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
    Fyy = abs(np.fft.fft(yy))
    guess_freq = abs(ff[np.argmax(Fyy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
    guess_amp = np.std(yy) * 2.**0.5
    guess_offset = np.mean(yy)
    guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset])

    def sinfunc(t, A, w, p, c):  return A * np.sin(w*t + p) + c
    popt, pcov = scipy.optimize.curve_fit(sinfunc, tt, yy, p0=guess)
    A, w, p, c = popt
    f = w/(2.*np.pi)
    fitfunc = lambda t: A * np.sin(w*t + p) + c
    return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1./f, "fitfunc": fitfunc, "maxcov": np.max(pcov), "rawres": (guess,popt,pcov)}

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

    spectr = np.fft.fftn(m)

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
    spectr = np.roll(spectr, shift[1])
    spectr = np.roll(spectr, shift[0]) / N

    return spectr, freq

def split_data(data):
    # data = data.tolist()
    s01, s02 = [], []
    chunk01 = data[:77]
    s01 = np.concatenate([s01,chunk01])
    for i in range(len(data) // 365 + 100):
        chunk1 = data[i * 365 + 265:i * 365 + 265 + 100 + 77]
        s01 = np.concatenate([s01,chunk1])
    for i in range(len(data) // 365):
        chunk2 = data[i * 365 + 77:i * 365 + 265]
        s02 = np.concatenate([s02,chunk2])
    s1 = np.array(s01)
    s2 = np.array(s02)
    return s1, s2

def calculate_length_of_day(jd):
    jd = np.array(jd)
    len_of_day = np.zeros((len(jd)-1)//2)
    for j in range(0,len(jd)-1,2):
        len_of_day[j//2] = jd[j+1] - jd[j]
    return len_of_day

def convert_localextr_to_stochastic_process(level: float,
                                            dates_input: np.ndarray,
                                            values_input: np.ndarray,
                                            file_name: str) -> None:
    filtered_values = np.array([1 if x >= level else 0 for x in values_input])
    indices = np.where(filtered_values == 1)[0]
    stoch_proc = pd.DataFrame({
        'dates': dates_input[indices],
        'values': filtered_values[indices]
    })

    stoch_proc.to_csv(file_name, header=False, index=False, sep=' ')

def resample_data(df, date_column, freq='ME'):
    df[date_column] = pd.to_datetime(df[date_column])
    df.set_index(date_column, inplace=True)
    df = df.resample(freq).mean()
    df.index = df.index.to_period('M').to_timestamp()
    return df

def read_climate_index(filepath):
    df = pd.read_csv(f'data/{filepath}', sep='\s+')
    df['dates'] = [convert_decimal_year(d) for d in df['dates']]
    df['dates'] = pd.to_datetime(df['dates'])
    df.set_index('dates', inplace=True)
    return df

df_rad = pd.read_csv('data/dataexport_20241204T120753.csv',sep=',')
dates = np.array(df_rad['Date'])
rad_val = np.array(df_rad['rad_val'])

df_rad_monthly = resample_data(df_rad, 'Date')

print(np.where(dates == '2023-01-01')[0], np.where(dates == '2024-07-14')[0])

dt = 0.00274
k = np.arange(0, len(dates))
k_m = np.arange(0, len(df_rad_monthly['rad_val']))

res = fit_sin(k, rad_val)
print(res)

sine_df = pd.DataFrame({
    'date': dates,
    'value': res['fitfunc'](k)
})

sine_df_monthly = resample_data(sine_df, 'date')

detrended_rad = rad_val - res['fitfunc'](k)
# detrended_rad_monthly = df_rad_monthly['rad_val'] - res['fitfunc'](k_m)

print(len(detrended_rad))

autocor = compute_acf_numpy(detrended_rad - np.mean(detrended_rad),
                            detrended_rad - np.mean(detrended_rad))

# 1. Два суб ряда
winter_dates, summer_dates = split_data(dates)
winter_data, summer_data = split_data(detrended_rad)

print(len(winter_data), len(summer_data))

# 2. Огибающая
omega_0 = 0.05
t = np.arange(-500, 500)
h = pantelleev_filter(t, omega_0)
max_indices = argrelextrema(detrended_rad, np.greater, order=10)[0]
min_indices = argrelextrema(detrended_rad, np.less, order=10)[0]
max_indices_abs = argrelextrema(np.abs(detrended_rad), np.greater, order=5)[0]
abs_envelope = np.interp(k, k[max_indices_abs], np.abs(detrended_rad)[max_indices_abs])
abs_envelope_conv = manual_convolution(abs_envelope, h)
upper_envelope = np.interp(k, k[max_indices], detrended_rad[max_indices])
lower_envelope = np.interp(k, k[min_indices], detrended_rad[min_indices])

sine_envelope_abs = fit_sin(k, abs_envelope)['fitfunc']

df_envelope = pd.DataFrame({
    'Dates': dates,
    'Values': abs_envelope - sine_envelope_abs(k)
})

df_envelope_monthly = resample_data(df_envelope, 'Dates')

# 3. Прилив
df_tide = pd.read_csv('data/LOD_zonal.dat',sep='\s+')

dates_tide = np.array(df_tide['year'])
dates_tide_slice = dates_tide[87658:118662]
tide_ut = np.array(df_tide['dut'])[np.where(dates_tide == 1940.00273)[0][0]:np.where(dates_tide == 2024.88464)[0][0]]
tide_lod = np.array(df_tide['dlod'])[np.where(dates_tide == 1940.00273)[0][0]:np.where(dates_tide == 2024.88464)[0][0]]
tide_omega = np.array(df_tide['omega'])[np.where(dates_tide == 1940.00273)[0][0]:np.where(dates_tide == 2024.88464)[0][0]]
tide_lod_temp = np.array(df_tide['dlod'])[np.where(dates_tide == 1996.84096)[0][0]:np.where(dates_tide == 2024.74812)[0][0]]
df_tide = pd.DataFrame({
    'dates': dates,
    'tide': tide_lod
})

df_tide_monthly = resample_data(df_tide, 'dates')

sine_tide = fit_sin(k, tide_lod)['fitfunc']

print(fit_sin(k, tide_omega))

true_omega = pd.read_csv("data/true_lod.dat", sep='\s+')

true_omega_values = np.array(-true_omega['lod'])[:23009]
true_omega_dates = np.array(dates_tide[95694:118703])
coeff_trend_true_omega = np.polyfit(true_omega_dates, true_omega_values, 1)
trend_true_omega = np.polyval(coeff_trend_true_omega, true_omega_dates)
detrend_true_omega = true_omega_values - trend_true_omega
max_true_omega_indices = argrelextrema(true_omega_values, np.greater, order=5)[0]
convert_localextr_to_stochastic_process(-1.0,
                                        true_omega_dates[max_true_omega_indices],
                                        true_omega_values[max_true_omega_indices],
                                        'true_omega_max.dat')

print(len(upper_envelope), detrended_rad)

max_tide_indices = argrelextrema(tide_omega, np.greater, order=5)[0]
min_tide_indices = argrelextrema(tide_omega, np.less, order=5)[0]
max_tide_envelope = np.array(df_envelope['Values'])[max_tide_indices]
min_tide_envelope = np.array(df_envelope['Values'])[min_tide_indices]
max_tide = np.array(df_tide['tide'])[max_tide_indices]
max_envelope_indices = argrelextrema(np.array(df_envelope['Values']), np.greater, order=5)[0]
max_envelope = np.array(df_envelope['Values'])[max_envelope_indices]

max_tide_rad = np.array(detrended_rad)[max_tide_indices]
min_tide_rad = np.array(detrended_rad)[min_tide_indices]

print(max_tide_indices.shape, max_tide_envelope.shape)

df_max_tide_envelope = pd.DataFrame({
    'dates': dates[max_tide_indices],
    'values': max_tide_envelope
})

df_min_tide_envelope = pd.DataFrame({
    'dates': dates[min_tide_indices],
    'values': min_tide_envelope
})

df_max_tide_envelope['dates'] = pd.to_datetime(df_max_tide_envelope['dates'])
df_max_tide_envelope.set_index('dates', inplace=True)

df_min_tide_envelope['dates'] = pd.to_datetime(df_min_tide_envelope['dates'])
df_min_tide_envelope.set_index('dates', inplace=True)

print(kstest(max_tide_envelope, min_tide_envelope))
print(ks_2samp(max_tide_envelope, min_tide_envelope))
print(mannwhitneyu(max_tide_envelope, min_tide_envelope))
print(wilcoxon(max_tide_envelope, min_tide_envelope[:-1]))

# 4. Осадки
precipitation = pd.read_csv('data/Atm_Temp_Osadki.dat', sep=' ')

dates_precip = np.array(precipitation['dates'])
dates_precip_tide = dates_tide[np.where(dates_tide == 1996.84096)[0][0]:np.where(dates_tide == 2024.74812)[0][0]]
precip_1 = np.array(precipitation['1'])
precip_2 = np.array(precipitation['2'])
precip_3 = np.array(precipitation['3'])

averaged_by_day_precip_1 = averaged_by_interval(precip_1, 8)
averaged_by_day_precip_2 = averaged_by_interval(precip_2, 8)
averaged_by_day_precip_3 = averaged_by_interval(precip_3, 8)

t_precip = np.arange(1996.84096, 2024.74812, 0.002735)
filtered_precip = manual_convolution(averaged_by_day_precip_1, h)

k_t = np.arange(0, len(averaged_by_day_precip_2))
sine_temp = fit_sin(k_t, averaged_by_day_precip_2)['fitfunc']

trendless_temp = np.array(averaged_by_day_precip_2 - sine_temp(k_t))
max_temp_indices = argrelextrema(trendless_temp, np.greater, order=5)[0]
max_temp = np.array([trendless_temp[i] if i in max_temp_indices else 0 for i in range(len(k_t))])

pressure = np.array(averaged_by_day_precip_1)
max_pressure_indices = argrelextrema(pressure, np.greater, order=5)[0]
max_pressure = np.array([pressure[i] if i in max_pressure_indices else 0 for i in range(len(k_t))])

h_shifted = np.roll(h, 13) + np.roll(h, -13)

pant_smooth = pd.read_csv('data/27-29pantsmoothT.dat', sep=' ')
pant_smooth_temp = np.append(np.array(pant_smooth.iloc[:, 3]), 2.0)
pant_smooth_temp_daily = averaged_by_interval(pant_smooth_temp, 8)

# 5. Длительность дня

length_of_day = pd.read_csv('data/sunsetrise_fixed.txt',sep='\s+')

JD_length_of_day = length_of_day['JD']
len_of_day = calculate_length_of_day(JD_length_of_day)

# 6. NAO, SOI, etc.
df_nao, df_soi, df_amo = read_climate_index('NAO.dat'), read_climate_index('soi.dat'), read_climate_index('amon.us.long.dat')

X = [df_envelope_monthly['Values']['1951-01-01':'2023-01-01'],
     df_nao['nao']['1951-01-01':'2023-01-01'],
     df_soi['soi']['1951-01-01':'2023-01-01'],
     df_amo['amo']['1951-01-01':'2023-01-01'],
     df_tide_monthly['tide']['1951-01-01':'2023-01-01']]
cov_matrix = np.cov(X)
std_diag = np.sqrt(np.diag(cov_matrix))
corr_matrix = cov_matrix / np.outer(std_diag, std_diag)

for row in corr_matrix:
    formatted_row = ["{:.2f}".format(elem) for elem in row]
    print(" ".join(formatted_row))

# 7. Построение всех спектров

specter_rad, omega_rad = ampl_fft(rad_val)
detrend_specter_rad, detrend_omega_rad = ampl_fft(detrended_rad)

winter_specter, winter_omega = ampl_fft(winter_data)
summer_specter, summer_omega = ampl_fft(summer_data)

envelope_specter, envelope_omega = ampl_fft(df_envelope)

specter_tide, omega_tide = ampl_fft(tide_lod)

specter_precip_1, omega_precip_1 = ampl_fft(averaged_by_day_precip_1)
specter_precip_2, omega_precip_2 = ampl_fft(averaged_by_day_precip_2 - sine_temp(k_t))
specter_precip_3, omega_precip_3 = ampl_fft(averaged_by_day_precip_3)

specter_filtered_precip, omega_filtered_precip = ampl_fft(filtered_precip)

specter_envelope, omega_envelope = ampl_fft(np.convolve(df_envelope_monthly['Values'] - np.mean(df_envelope_monthly['Values']),
                                                        df_envelope_monthly['Values'] - np.mean(df_envelope_monthly['Values']),
                                                        'same'))

filtered_temp_specter = specter_precip_2 * np.append(np.pad(h_shifted, 4596, 'constant'), 0)
filtered_temp = np.fft.ifft(filtered_temp_specter)

# 8. Авто/Кросс-ковариационные функции

autocor_envelope = compute_acf_numpy(np.array(df_envelope['Values']), np.array(df_envelope['Values']))
autocor_envelope_monthly = compute_acf_numpy(df_envelope_monthly['Values'], df_envelope_monthly['Values'])
autocor_envelope_specter, autocor_envelope_omega = ampl_fft(autocor_envelope)
crosscor_tide_envelope = compute_acf_numpy(tide_lod, df_envelope['Values'])
crosscor_tide_rad = compute_acf_numpy(tide_lod, rad_val - res['fitfunc'](k))
crosscor_tide_temp = compute_acf_numpy(tide_lod_temp, averaged_by_day_precip_2 - sine_temp(k_t))
crosscor_tide_envelope_specter, crosscor_tide_envelope_omega = ampl_fft(crosscor_tide_envelope)
crosscor_tide_rad_specter, crosscor_tide_rad_omega = ampl_fft(crosscor_tide_rad)
# crosscor_tide_temp = compute_acf_numpy(tide_lod[t_precip], averaged_by_day_precip_2)
crosscor_tide_temp_specter, crosscor_tide_temp_omega = ampl_fft(crosscor_tide_temp)

# 9. Параметр Хёрста?

# periods = np.array(abs(1/detrend_omega_rad))
# hurst_poly_coeffs = np.polyfit(periods[2000:10000],
#                         np.abs(detrend_specter_rad[2000:10000]), 1)
# hurst_poly_val = np.polyval(hurst_poly_coeffs[2000:10000], periods[2000:10000])

# 10. Любушин

intensities = pd.read_csv("data/Intensity_Shares_cleaned.dat", sep='\s+', header=None)

plt.figure(1)
plt.plot(df_rad.index, df_rad['rad_val'])
# plt.plot(sine_df)
plt.xlabel('Даты')
plt.ylabel('Мера солнечного излучения, Вт/м2')
plt.title("ПСИ в городе Базель, Швейцария")
plt.grid()

plt.figure(2)
plt.plot(df_tide.index, np.abs(df_rad['rad_val'] - sine_df['value']), label='Данные солнечного излучения без годового тренда')
# plt.plot(df_tide.index, df_tide['tide'])
# plt.plot(500*(sine_tide(k)))
# plt.plot(dates[30314:30874], 7800*len_of_day-2000, label='Продолжительность дня')
plt.plot(df_tide.index, abs_envelope, label='Огибающая')
# plt.xticks([dates[120*i] for i in range(258)])
plt.legend(loc='best')
plt.grid()

plt.figure(3)
plt.plot(res['period']/omega_rad[len(omega_rad)//2:], np.abs(specter_rad[len(specter_rad)//2:]))
plt.grid()

plt.figure(4)
plt.plot(365.25/omega_rad, np.abs(specter_rad))
plt.xscale("log")
plt.grid()

plt.figure(5)
# plt.plot(1/detrend_omega_rad, np.abs(detrend_specter_rad), label='Прилив солнечного излучения')
plt.plot(res['period']/omega_tide, np.abs(specter_tide), label='omega из LOD_zonal.dat')
plt.title('Спектры прилива солнечного излучения и лунного прилива')
plt.xlabel('Циклов в год')
plt.xscale('log')
plt.legend(loc='best')
plt.grid()

plt.figure(6)
plt.plot(1/detrend_omega_rad, np.abs(detrend_specter_rad))
# plt.plot(periods[2000:10000], hurst_poly_val)
# plt.xscale("log")
plt.grid()

plt.figure(7)
plt.plot(autocor, linewidth=0.75)
plt.xlabel("Временной сдвиг в днях")
plt.title("Автокорреляционная функция для ПСИ без тренда")
plt.grid()

# plt.figure(8)
# plt.plot(winter_dates, winter_data, linewidth=0.75)
# plt.xticks([winter_dates[60*i] for i in range(250)])
# plt.title('"Зимний" суб ряд, с небольшим разбросом (между 23 сентября и 19 марта)')
# plt.xlabel('Даты')
# plt.grid()

# plt.figure(9)
# plt.plot(summer_dates, summer_data, linewidth=0.75)
# plt.xticks([summer_dates[60*i] for i in range(264)])
# plt.title('"Летний" суб ряд, с большим разбросом (между 19 марта и 23 сентября)')
# plt.xlabel('Даты')
# plt.grid()

# plt.figure(10)
# plt.plot(winter_omega[:len(winter_omega)//2], np.abs(winter_specter[:len(winter_omega)//2]), linewidth=0.75)
# plt.title('Спектр "Зимнего" суб ряда')
# plt.xlabel('Циклов в год')
# plt.grid()

# plt.figure(11)
# plt.plot(summer_omega[:len(summer_omega)//2], np.abs(summer_specter[:len(summer_omega)//2]), linewidth=0.75)
# plt.title('Спектр "Летнего" суб ряда')
# plt.xlabel('Циклов в год')
# plt.grid()

plt.figure(12)
plt.plot(df_envelope.index, np.abs(detrended_rad), label='Модуль от ряда без годового тренда')
plt.plot(df_envelope.index, abs_envelope_conv, label='Огибающая')
# plt.plot(dates, lower_envelope, label='Нижняя огибающая')
plt.legend(loc='best')
# plt.xticks([dates[90*i] for i in range(345)])
plt.xlabel("Даты")
plt.ylabel("Модуль от данных ПСИ без годового тренда")
plt.title('Данные вместе с огибающими')
plt.grid()

# plt.figure(13)
# plt.plot(dates, np.abs(detrended_rad) - abs_envelope)
# plt.title('Данные без годового тренда и без верхней огибающей')
# plt.grid()

# plt.figure(14)
# plt.plot(dates, detrended_rad - lower_envelope)
# plt.title('Данные без годового тренда и без нижней огибающей')
# plt.grid()

# plt.figure(15)
# plt.plot(dates, np.abs(detrended_rad))
# plt.plot(abs_envelope)
# plt.grid()

plt.figure(16)
plt.plot(dates_precip, precip_1)
# plt.plot(dates_precip, precip_2)
# plt.plot(dates_precip, precip_3)
plt.grid()

plt.figure(17)
# plt.plot(dates_tide[np.where(dates_tide == 1996.84096)[0][0]:np.where(dates_tide == 2024.74812)[0][0]], averaged_by_day_precip_1)
plt.plot(dates_tide[np.where(dates_tide == 1996.84096)[0][0]:np.where(dates_tide == 2024.74812)[0][0]], averaged_by_day_precip_2 - sine_temp(k_t))
# plt.plot(dates_tide[np.where(dates_tide == 1996.84096)[0][0]:np.where(dates_tide == 2024.74812)[0][0]], averaged_by_day_precip_3)
plt.grid()

plt.figure(18)
plt.plot(res['period']/omega_precip_2, abs(specter_precip_2))
# plt.plot(t, h_shifted)
# plt.plot(omega_tide[len(omega_tide)//2:], np.abs(specter_tide)[len(omega_tide)//2:])
plt.scatter(27.1, 0.1694, marker='^', color='r', zorder=5)
plt.scatter(29.2043, 0.2096, marker='^', color='orange', zorder=5)
plt.text(27.1017 - 25.3, 0.1694, "Сидерический период Луны ~27 дней")
plt.text(29.2043 - 27.3, 0.2096, "Синодический период Луны ~29 дней")
plt.xscale('log')
plt.title("Периодограмма данных температуры на Камчатке без годового цикла")
plt.xlabel('Периоды в днях')
plt.grid()

plt.figure(19)
plt.plot(dates_precip_tide, averaged_by_day_precip_2)
# plt.plot(dates_tide[np.where(dates_tide == 1996.84096)[0][0]:np.where(dates_tide == 2024.74812)[0][0]], pant_smooth_temp_daily)
plt.plot(dates_precip_tide, sine_temp(k_t))
plt.title("Ряд температуры на Камчатке с подобранной годовой гармоникой")
plt.xlabel("Даты")
plt.ylabel("Температура, °С")
plt.grid()

# plt.figure(20)
# plt.plot(omega_filtered_precip[len(omega_filtered_precip)//2:], np.abs(specter_filtered_precip)[len(omega_filtered_precip)//2:])
# plt.grid()

plt.figure(21)
plt.plot(h)

plt.figure(22)
plt.plot(res['period']/envelope_omega, np.abs(envelope_specter))
# plt.plot(1/omega_rad, np.abs(specter_rad))
plt.xscale("log")

plt.figure(23)
plt.subplot(5,1,1)
plt.plot(df_envelope_monthly.index, np.array(df_envelope_monthly['Values']), label='Огибающая')
# plt.plot(df_envelope.index, manual_convolution(np.array(df_envelope['Values']), h), label='Отфильтрованная огибающая')
plt.legend(loc='best')
plt.title("Графики данных")
plt.grid()
plt.subplot(5,1,2)
plt.plot(df_nao.index, df_nao['nao'], label='NAO')
plt.legend(loc='best')
plt.grid()
plt.subplot(5,1,3)
plt.plot(df_soi.index, df_soi['soi'], label='SOI')
plt.legend(loc='best')
plt.grid()
plt.subplot(5,1,4)
plt.plot(df_amo.index, df_amo['amo'], label='AMO')
plt.legend(loc='best')
plt.grid()
# plt.plot(df_max_lod_envelope.index, df_max_lod_envelope['values'], label='lod_max')
plt.subplot(5,1,5)
plt.plot(df_tide.index, df_tide['tide'], label='Прилив')
plt.legend(loc='best')
plt.xlabel("Даты")
plt.grid()

plt.figure(24)
plt.plot(autocor_envelope_monthly)
plt.grid()

plt.figure(25)
plt.plot(365.25/omega_envelope, np.abs(specter_envelope))
plt.xscale('log')
plt.grid()

plt.figure(26)
plt.subplot(2,2,1)
plt.plot(df_envelope.index, df_envelope['Values'], label='Envelope')
plt.plot(df_max_tide_envelope.index, df_max_tide_envelope['values'], label='lod_max')
plt.legend(loc='best')
plt.grid()

plt.subplot(2,2,2)
plt.plot(df_envelope.index, df_envelope['Values'], label='Envelope')
plt.plot(df_min_tide_envelope.index, df_min_tide_envelope['values'], label='lod_min')
plt.legend(loc='best')
plt.grid()

plt.subplot(2,2,3)
histplot(df_max_tide_envelope['values'])
plt.title('Max tide in envelope')

plt.subplot(2,2,4)
histplot(df_min_tide_envelope['values'])
plt.title('Min tide in envelope')

plt.figure(27)
plt.plot(crosscor_tide_envelope)
plt.grid()

plt.figure(28)
plt.plot(res['period']/crosscor_tide_envelope_omega[len(crosscor_tide_envelope_omega)//2:], np.abs(crosscor_tide_envelope_specter[len(crosscor_tide_envelope_omega)//2:]))
# plt.scatter(27.1, 0.256, marker='^', color='r', zorder=5)
# plt.text(27.1017 - 25.8, 0.256, "Сидерический период Луны ~27 дней")
plt.title("Спектральная мощность сигнала ККФ (кросс-ковариационной функции) от прилива и огибающей")
plt.xscale('log')
plt.xlabel("Периоды в днях")
plt.grid()

plt.figure(29)
plt.plot(res['period']/crosscor_tide_rad_omega[len(crosscor_tide_rad_omega)//2:], np.abs(crosscor_tide_rad_specter[len(crosscor_tide_rad_omega)//2:]))
plt.title("Спектральная мощность сигнала ККФ (кросс-ковариационной функции) от прилива и данных ПСИ")
plt.xscale('log')
plt.xlabel("Периоды в годах")
plt.grid()

# plt.figure(30)
# plt.plot(1/crosscor_tide_temp_omega[len(crosscor_tide_temp_omega)//2:], np.abs(crosscor_tide_temp_specter[len(crosscor_tide_temp_omega)//2:]))
# plt.title("Спектральная мощность сигнала ККФ (кросс-ковариационной функции) от прилива и температурой на Камчатке")
# plt.xscale('log')
# plt.xlabel("Периоды в годах")
# plt.grid()

plt.figure(31)
ax = heatmap(corr_matrix, annot=True, fmt='.4f')
ax.set_yticklabels(['Огибающая','NAO','SOI','AMO','Прилив'])
ax.set_xticklabels(['Огибающая','NAO','SOI','AMO','Прилив'])
plt.title('Корреляционная матрица для индексов, огибающей и прилива')

plt.figure(32)
plt.plot(filtered_temp)
plt.plot()

plt.figure(33)
plt.plot(res['period']/crosscor_tide_temp_omega[len(crosscor_tide_temp_omega)//2:], np.abs(crosscor_tide_temp_specter[len(crosscor_tide_temp_omega)//2:]))
plt.xscale('log')
plt.grid()

plt.figure(34)
plt.subplot(2,2,1)
plt.plot(df_envelope.index, detrended_rad, label='Envelope')
plt.plot(df_max_tide_envelope.index, max_tide_rad, label='lod_max')
plt.legend(loc='best')
plt.grid()

plt.subplot(2,2,2)
plt.plot(df_envelope.index, detrended_rad, label='Envelope')
plt.plot(df_min_tide_envelope.index, min_tide_rad, label='lod_min')
plt.legend(loc='best')
plt.grid()

plt.subplot(2,2,3)
histplot(max_tide_rad)
plt.title('Max tide in envelope')

plt.subplot(2,2,4)
histplot(min_tide_rad)
plt.title('Min tide in envelope')

plt.figure(35)
plt.subplot(3,2,1)
plt.plot(intensities[0], intensities[3])
plt.title('NAO\n\nСлучайная доля интенсивности')
plt.xticks([])

plt.subplot(3,2,3)
plt.plot(intensities[0], intensities[4])
plt.title('Доля самовозбуждения')
plt.xticks([])

plt.subplot(3,2,5)
plt.plot(intensities[0], intensities[5])
plt.title('Доля действия процесса огибающей на NAO')

plt.subplot(3,2,2)
plt.plot(intensities[0], intensities[6])
plt.title('Огибающая\n\nСлучайная доля интенсивности')
plt.xticks([])

plt.subplot(3,2,4)
plt.plot(intensities[0], intensities[7])
plt.title('Доля самовозбуждения')
plt.xticks([])

plt.subplot(3,2,6)
plt.plot(intensities[0], intensities[8])
plt.title('Доля действия процесса NAO на огибающую')

plt.show()

