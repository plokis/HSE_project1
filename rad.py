import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
from scipy.signal import argrelextrema
from scipy.stats import kstest, ks_2samp, mannwhitneyu, wilcoxon
from seaborn import histplot, heatmap

# Импорт собственных модулей
from analysis.utils import averaged_by_interval, calculate_length_of_day, split_data
from analysis.filters import moving_average, pantelleev_filter, manual_convolution
from analysis.spectral import compute_acf_numpy, ampl_fft
from analysis.fitting import fit_sin
from analysis.preprocessing import resample_data, read_climate_index, convert_localextr_to_stochastic_process
from analysis.plotting import (
    plot_rad_timeseries, plot_detrended_with_envelope, plot_filter_shape, plot_autocorrelation,
    plot_rad_spectrum_half, plot_rad_spectrum_log, plot_tide_spectrum, plot_detrended_rad_spectrum,
    plot_envelope_spectrum, plot_envelope_autocorr, plot_envelope_power, plot_ccf_spectrum,
    plot_ccf_spectrum_rad, plot_ccf_spectrum_temp, plot_indices_panel, plot_precip_raw,
    plot_temp_detrended, plot_temp_periodogram, plot_temp_with_fit, plot_filtered_temp,
    plot_envelope_over_lod, plot_detrended_vs_lod, plot_correlation_matrix, plot_crosscorr_series,
    plot_intensities_panel, show_all
)

# ----------------------
# 1. ДАННЫЕ ПСИ
# ----------------------
df_rad = pd.read_csv('data/dataexport_20241204T120753.csv', sep=',')
dates = np.array(df_rad['Date'])
rad_val = np.array(df_rad['rad_val'])
df_rad_monthly = resample_data(df_rad, 'Date')

k = np.arange(0, len(dates))
res = fit_sin(k, rad_val)
sine_df = pd.DataFrame({'date': dates, 'value': res['fitfunc'](k)})
sine_df_monthly = resample_data(sine_df, 'date')
detrended_rad = rad_val - res['fitfunc'](k)
# detrended_rad_monthly = df_rad_monthly['rad_val'] - res['fitfunc'](k_m)

print(len(detrended_rad))

# Автокорреляция ПСИ
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
                                        'data/true_omega_max.dat')

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

# ----------------------
# 7. ГРАФИКИ
# ----------------------
# 1
plot_rad_timeseries(df_rad, sine_df, fig=1)
# 2 и 12 (вторая — более детальная с огибающей свёрткой)
plot_detrended_with_envelope(df_envelope.index, np.abs(detrended_rad), abs_envelope_conv, fig=2)
plot_detrended_with_envelope(df_envelope.index, np.abs(detrended_rad), abs_envelope_conv, fig=12,
                             title='Данные вместе с огибающими')

# 3,4,5,6
half = slice(len(omega_rad)//2, None)
plot_rad_spectrum_half(res['period'] / omega_rad[half], specter_rad[half], fig=3)
plot_rad_spectrum_log(365.25/omega_rad, specter_rad, fig=4, title=None)
plot_tide_spectrum(res['period']/omega_tide, specter_tide, fig=5,
                   title='Спектры прилива солнечного излучения и лунного прилива')
plot_detrended_rad_spectrum(1/detrend_omega_rad, detrend_specter_rad, fig=6)

# 7
plot_autocorrelation(autocor, fig=7)

# 16–19
plot_precip_raw(dates_precip, precip_1, precip_2=None, precip_3=None, fig=16)
plot_temp_detrended(dates_precip_tide, (averaged_by_day_precip_2 - sine_temp(k_t)), fig=17,
                    title="Температура на Камчатке без годового цикла")
plot_temp_periodogram(res['period']/omega_precip_2, specter_precip_2, fig=18,
                      title="Периодограмма данных температуры на Камчатке без годового цикла")
plot_temp_with_fit(dates_precip_tide, averaged_by_day_precip_2, sine_temp(k_t), fig=19,
                   title="Температура на Камчатке с подобранной годовой гармоникой")

# 21,22,24,25
plot_filter_shape(h, fig=21)
plot_envelope_spectrum(res['period']/envelope_omega, envelope_specter, fig=22)
plot_envelope_autocorr(autocor_envelope_monthly, fig=24)
# plot_envelope_power(365.25/np.array(envelope_omega), np.convolve(df_envelope_monthly['Values']-np.mean(df_envelope_monthly['Values']),
#                                                                  df_envelope_monthly['Values']-np.mean(df_envelope_monthly['Values']),
#                                                                  'full'), fig=25)

# 23: панель индексов
plot_indices_panel(df_envelope_monthly, df_nao, df_soi, df_amo, df_tide, fig=23)

# 26: наложения и гистограммы по LOD
plot_envelope_over_lod(df_envelope.set_index(pd.to_datetime(df_envelope.index)),
                       df_max_tide_envelope, df_min_tide_envelope, fig=26)

# 27–29: ККФ и спектры
plot_crosscorr_series(crosscor_tide_envelope, fig=27, title="ККФ: прилив и огибающая")
plot_ccf_spectrum(res['period']/crosscor_tide_envelope_omega[len(crosscor_tide_envelope_omega)//2:],
                  crosscor_tide_envelope_specter[len(crosscor_tide_envelope_omega)//2:], fig=28,
                  title="Спектральная мощность ККФ: прилив vs огибающая", xlabel="Периоды в днях")
plot_ccf_spectrum_rad(res['period']/crosscor_tide_rad_omega[len(crosscor_tide_rad_omega)//2:],
                      crosscor_tide_rad_specter[len(crosscor_tide_rad_omega)//2:], fig=29,
                      title="Спектральная мощность ККФ: прилив vs ПСИ", xlabel="Периоды в годах")

# 31: Корреляционная матрица
plot_correlation_matrix(corr_matrix, labels=['Огибающая','NAO','SOI','AMO','Прилив'], fig=31)

# 32: отфильтрованная температура
plot_filtered_temp(filtered_temp, fig=32)

# 33: спектр ККФ с температурой
plot_ccf_spectrum_temp(res['period']/crosscor_tide_temp_omega[len(crosscor_tide_temp_omega)//2:],
                       crosscor_tide_temp_specter[len(crosscor_tide_temp_omega)//2:], fig=33,
                       title="Спектральная мощность ККФ: прилив vs температура")

# 34: ПСИ без тренда vs LOD экстремумы + гистограммы
plot_detrended_vs_lod(pd.to_datetime(df_envelope.index),
                      detrended_rad,
                      pd.to_datetime(dates[max_tide_indices]), max_tide_rad,
                      pd.to_datetime(dates[min_tide_indices]), min_tide_rad, fig=34)

# 35: панель интенсивностей
plot_intensities_panel(intensities, fig=35)

# финальный вывод
show_all()

