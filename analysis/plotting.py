import numpy as np
import matplotlib.pyplot as plt
from seaborn import histplot, heatmap

# ---------- БАЗОВЫЕ ВСПОМОГАТЕЛЬНЫЕ ----------

def _new_fig(fig=None, title=None):
    if fig is not None:
        plt.figure(fig)
    else:
        plt.figure()
    if title:
        plt.title(title)

def show_all():
    plt.show()

# ---------- ВРЕМЕННЫЕ РЯДЫ ----------

def plot_rad_timeseries(df_rad, sine_df=None, fig=1, title="ПСИ в городе Базель, Швейцария"):
    _new_fig(fig, title)
    plt.plot(df_rad.index, df_rad['rad_val'], label='Данные ПСИ')
    if sine_df is not None:
        plt.plot(sine_df.index, sine_df['value'], label='Годовой тренд (синус)')
    plt.xlabel('Даты'); plt.ylabel('Вт/м²'); plt.grid(); plt.legend()

def plot_detrended_with_envelope(dates, detrended_abs, envelope, fig=2,
                                 title="ПСИ без годового тренда и огибающая"):
    _new_fig(fig, title)
    plt.plot(dates, detrended_abs, label='|ПСИ без тренда|')
    plt.plot(dates, envelope, label='Огибающая')
    plt.xlabel('Даты'); plt.grid(); plt.legend(loc='best')

def plot_filter_shape(h, fig=21, title="Импульсная характеристика фильтра (Pantelleev)"):
    _new_fig(fig, title)
    plt.plot(h); plt.grid()

# ---------- СПЕКТРЫ / ПЕРИОДОГРАММЫ ----------

def plot_spectrum_by_period(periods, amplitude, fig=None, title=None,
                            xlabel="Периоды", logx=False, annotate=None):
    _new_fig(fig, title)
    plt.plot(periods, np.abs(amplitude))
    if logx: plt.xscale("log")
    if annotate:
        for (x, y, text) in annotate:
            plt.scatter(x, y, marker='^', zorder=5)
            plt.text(x, y, text)
    plt.xlabel(xlabel); plt.grid()

def plot_two_axes_spectrum(x, y, fig=None, title=None, xlabel=None, logx=False):
    _new_fig(fig, title)
    plt.plot(x, np.abs(y))
    if logx: plt.xscale("log")
    if xlabel: plt.xlabel(xlabel)
    plt.grid()

# ---------- АВТО-/КРОСС-КОВАРИАЦИИ И ИХ СПЕКТРЫ ----------

def plot_autocorrelation(acf, fig=7, title="Автокорреляционная функция для ПСИ без тренда"):
    _new_fig(fig, title)
    plt.plot(acf, linewidth=0.75)
    plt.xlabel("Сдвиг (дни)"); plt.grid()

def plot_crosscorr_series(ccf, fig=27, title="ККФ: прилив vs огибающая"):
    _new_fig(fig, title)
    plt.plot(ccf); plt.grid()

# ---------- ПАНЕЛИ И МАТРИЦЫ ----------

def plot_correlation_matrix(corr_matrix, labels, fig=31, title='Корреляционная матрица'):
    _new_fig(fig, title)
    ax = heatmap(corr_matrix, annot=True, fmt='.4f')
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)

def plot_indices_panel(df_envelope_monthly, df_nao, df_soi, df_amo, df_tide,
                       fig=23, title="Графики данных"):
    _new_fig(fig, title)
    plt.subplot(5,1,1)
    plt.plot(df_envelope_monthly.index, df_envelope_monthly['Values'], label='Огибающая')
    plt.legend(loc='best'); plt.grid()
    plt.subplot(5,1,2)
    plt.plot(df_nao.index, df_nao['nao'], label='NAO'); plt.legend(loc='best'); plt.grid()
    plt.subplot(5,1,3)
    plt.plot(df_soi.index, df_soi['soi'], label='SOI'); plt.legend(loc='best'); plt.grid()
    plt.subplot(5,1,4)
    plt.plot(df_amo.index, df_amo['amo'], label='AMO'); plt.legend(loc='best'); plt.grid()
    plt.subplot(5,1,5)
    plt.plot(df_tide.index, df_tide['tide'], label='Прилив'); plt.legend(loc='best'); plt.xlabel("Даты"); plt.grid()

# ---------- ОСАДКИ/ТЕМПЕРАТУРА ----------

def plot_precip_raw(dates_precip, precip_1, precip_2=None, precip_3=None, fig=16):
    _new_fig(fig, "Осадки/давление (сырые ряды)")
    plt.plot(dates_precip, precip_1, label='1')
    if precip_2 is not None: plt.plot(dates_precip, precip_2, label='2')
    if precip_3 is not None: plt.plot(dates_precip, precip_3, label='3')
    plt.grid(); plt.legend()

def plot_temp_detrended(dates, temp_detrended, fig=17, title="Температура без годового цикла"):
    _new_fig(fig, title)
    plt.plot(dates, temp_detrended); plt.grid()

def plot_temp_periodogram(periods, amplitude, fig=18, title="Периодограмма температуры (без годового цикла)"):
    _new_fig(fig, title)
    plt.plot(periods, np.abs(amplitude))
    # метки лунных периодов
    plt.scatter(27.1, 0.1694, marker='^', zorder=5)
    plt.scatter(29.2043, 0.2096, marker='^', zorder=5)
    plt.text(27.1017 - 25.3, 0.1694, "Сидерический период Луны ~27 дней")
    plt.text(29.2043 - 27.3, 0.2096, "Синодический период Луны ~29 дней")
    plt.xscale('log'); plt.xlabel('Периоды в днях'); plt.grid()

def plot_temp_with_fit(dates, temp, fit, fig=19, title="Температура с подобранной годовой гармоникой"):
    _new_fig(fig, title)
    plt.plot(dates, temp, label='Темп. ряд')
    plt.plot(dates, fit, label='Годовая гармоника')
    plt.xlabel("Даты"); plt.ylabel("°C"); plt.grid(); plt.legend()

def plot_filtered_temp(filtered_temp, fig=32, title="Отфильтрованный температурный ряд"):
    _new_fig(fig, title)
    plt.plot(filtered_temp); plt.grid()

# ---------- ОГИБАЮЩАЯ vs LOD (наложения + гистограммы) ----------

def plot_envelope_over_lod(df_envelope, df_max_tide_envelope, df_min_tide_envelope, fig=26):
    _new_fig(fig, "Огибающая vs LOD (экстремумы)")
    plt.subplot(2,2,1)
    plt.plot(df_envelope.index, df_envelope['Values'], label='Envelope')
    plt.plot(df_max_tide_envelope.index, df_max_tide_envelope['values'], label='lod_max')
    plt.legend(loc='best'); plt.grid()
    plt.subplot(2,2,2)
    plt.plot(df_envelope.index, df_envelope['Values'], label='Envelope')
    plt.plot(df_min_tide_envelope.index, df_min_tide_envelope['values'], label='lod_min')
    plt.legend(loc='best'); plt.grid()
    plt.subplot(2,2,3)
    histplot(df_max_tide_envelope['values'])
    plt.title('Max tide in envelope')
    plt.subplot(2,2,4)
    histplot(df_min_tide_envelope['values'])
    plt.title('Min tide in envelope')

def plot_detrended_vs_lod(dates, detrended_rad, max_tide_dates, max_tide_rad,
                          min_tide_dates, min_tide_rad, fig=34):
    _new_fig(fig, "ПСИ без тренда vs LOD-экстремумы")
    plt.subplot(2,2,1)
    plt.plot(dates, detrended_rad, label='ПСИ без тренда')
    plt.plot(max_tide_dates, max_tide_rad, label='lod_max'); plt.legend(loc='best'); plt.grid()
    plt.subplot(2,2,2)
    plt.plot(dates, detrended_rad, label='ПСИ без тренда')
    plt.plot(min_tide_dates, min_tide_rad, label='lod_min'); plt.legend(loc='best'); plt.grid()
    plt.subplot(2,2,3)
    histplot(max_tide_rad); plt.title('Max tide in envelope')
    plt.subplot(2,2,4)
    histplot(min_tide_rad); plt.title('Min tide in envelope')

# ---------- СПЕКТРЫ ДЛЯ ОТДЕЛЬНЫХ СЛУЧАЕВ ----------

def plot_rad_spectrum_half(periods, spec_half, fig=3, title=None):
    plot_two_axes_spectrum(periods, spec_half, fig=fig, title=title, xlabel=None, logx=False)

def plot_rad_spectrum_log(periods, spec, fig=4, title=None):
    plot_two_axes_spectrum(periods, spec, fig=fig, title=title, xlabel="Периоды в годах", logx=True)

def plot_tide_spectrum(periods, spec, fig=5, title='Спектры прилива'):
    plot_two_axes_spectrum(periods, spec, fig=fig, title=title, xlabel='Циклов в год', logx=True)

def plot_detrended_rad_spectrum(periods, spec, fig=6, title=None):
    plot_two_axes_spectrum(periods, spec, fig=fig, title=title, xlabel=None, logx=False)

def plot_envelope_spectrum(periods, spec, fig=22, title=None):
    plot_two_axes_spectrum(periods, spec, fig=fig, title=title, xlabel=None, logx=True)

def plot_envelope_autocorr(acf_monthly, fig=24, title=None):
    _new_fig(fig, title)
    plt.plot(acf_monthly); plt.grid()

def plot_envelope_power(periods, power, fig=25, title=None):
    plot_two_axes_spectrum(periods, power, fig=fig, title=title, xlabel=None, logx=True)

def plot_ccf_spectrum(periods, power, fig=28, title=None, xlabel="Периоды в днях"):
    _new_fig(fig, title)
    plt.plot(periods, np.abs(power))
    plt.xscale('log'); plt.xlabel(xlabel); plt.grid()

def plot_ccf_spectrum_rad(periods, power, fig=29, title=None, xlabel="Периоды в годах"):
    _new_fig(fig, title)
    plt.plot(periods, np.abs(power))
    plt.xscale('log'); plt.xlabel(xlabel); plt.grid()

def plot_ccf_spectrum_temp(periods, power, fig=33, title=None):
    _new_fig(fig, title)
    plt.plot(periods, np.abs(power))
    plt.xscale('log'); plt.grid()

# ---------- ПАНЕЛЬ ИНТЕНСИВНОСТЕЙ ----------

def plot_intensities_panel(intensities_df, fig=35):
    _new_fig(fig, "Интенсивности (Любушин)")
    plt.subplot(3,2,1)
    plt.plot(intensities_df[0], intensities_df[3])
    plt.title('NAO\n\nСлучайная доля интенсивности'); plt.xticks([])
    plt.subplot(3,2,3)
    plt.plot(intensities_df[0], intensities_df[4])
    plt.title('Доля самовозбуждения'); plt.xticks([])
    plt.subplot(3,2,5)
    plt.plot(intensities_df[0], intensities_df[5])
    plt.title('Доля действия огибающей на NAO')
    plt.subplot(3,2,2)
    plt.plot(intensities_df[0], intensities_df[6])
    plt.title('Огибающая\n\nСлучайная доля интенсивности'); plt.xticks([])
    plt.subplot(3,2,4)
    plt.plot(intensities_df[0], intensities_df[7])
    plt.title('Доля самовозбуждения'); plt.xticks([])
    plt.subplot(3,2,6)
    plt.plot(intensities_df[0], intensities_df[8])
    plt.title('Доля действия NAO на огибающую')