import pandas as pd
import numpy as np


def convert_decimal_year(decimal_year):
    year = int(decimal_year)
    month = int(np.ceil((decimal_year - year + 0.01) * 12))
    month = max(1, min(month, 12))
    return f"{year:04d}-{month:02d}"


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