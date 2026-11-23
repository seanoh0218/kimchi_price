import warnings
warnings.filterwarnings(action='ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import koreanize_matplotlib

from pathlib import Path

import preprocessing
import analysis as anal
import modeling


BASE = Path(__file__).resolve().parent

def main():
    # 전처리
    print('===전처리===')
    try:
        df = pd.read_csv('data_lag_30.csv')
    except FileNotFoundError:
        preprocessing.preprocess()
        df = pd.read_csv('data_lag_30.csv')

    print(df.head())
    print(df.describe())
    print('===전처리 끝===')

    # 데이터 분석
    print('===데이터 분석===')
    df_0 = pd.read_csv('data/data.csv').drop(columns=['temp_min_anom', 'temp_max_anom'])
    df_0['date'] = pd.to_datetime(df_0['date'])

    df_30 = pd.read_csv('data/data_lag_30.csv').drop(columns=['temp_min_anom_30', 'temp_max_anom_30'])
    df_30['date'] = pd.to_datetime(df_30['date'])

    df_60 = pd.read_csv('data/data_lag_60.csv').drop(columns=['temp_min_anom_60', 'temp_max_anom_60'])
    df_60['date'] = pd.to_datetime(df_60['date'])

    df_90 = pd.read_csv('data/data_lag_90.csv').drop(columns=['temp_min_anom_90', 'temp_max_anom_90'])
    df_90['date'] = pd.to_datetime(df_90['date'])

    anal.plot_scatter_all_data(df_0, 'scatter_0')
    anal.plot_scatter_all_data(df_30, 'scatter_30')
    anal.plot_scatter_all_data(df_60, 'scatter_60')
    anal.plot_scatter_all_data(df_90, 'scatter_90')

    anal.plot_heatmap_all_data(df_0, 'heatmap_0')
    anal.plot_heatmap_all_data(df_30, 'heatmap_30')
    anal.plot_heatmap_all_data(df_60, 'heatmap_60')
    anal.plot_heatmap_all_data(df_90, 'heatmap_90')
    print('===데이터 분석 끝===')

    # 모델 학습 및 평가
    combos = {
        '전체': ['temp_avg_anom_30', 'rainfall_anom_30', 'wind_avg_anom_30', 'humidity_avg_anom_30', 'sunshine_sum_anom_30'],
        '기온+풍속': ['temp_avg_anom_30', 'wind_avg_anom_30'],
        '기온+풍속+습도': ['temp_avg_anom_30', 'wind_avg_anom_30', 'humidity_avg_anom_30'],
        '일조 제외': ['temp_avg_anom_30', 'rainfall_anom_30', 'wind_avg_anom_30', 'humidity_avg_anom_30'],
        '기온만': ['temp_avg_anom_30'],
    }

    # TODO
    for name, cols in combos.items():
        model = LinearRegression().fit(train_data[cols], train_data['price'])
        y_pred = model.predict(test_data[cols])
        r2 = r2_score(test_data['price'], y_pred)
        rmse = np.sqrt(mean_squared_error(test_data['price'], y_pred))
        print(f'{name:10s}  R2 = {r2:6.3f}  RMSE = {rmse:7.1f}')


if __name__ == '__main__':
    main()
