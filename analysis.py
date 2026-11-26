import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import koreanize_matplotlib

from pathlib import Path


BASE = Path(__file__).resolve().parent

def plot_scatter_all_data(data: pd.DataFrame, fname: str, base_path: Path = BASE):
    """모든 데이터 산점도 그리기"""
    sns.pairplot(data)
    g = sns.pairplot(data)
    g.figure.suptitle('데이터 간 상관관계', y=1.02)
    plt.savefig(base_path / f'figures/{fname}.svg', bbox_inches='tight')
    plt.close()


def plot_heatmap_all_data(data: pd.DataFrame, fname: str, base_path: Path = BASE):
    """히트맵 그래프를 활용하여 속성 간 상관관계 시각화"""
    corr = data.corr()
    plt.figure(figsize=(12, 10))

    sns.heatmap(
        corr,
        annot=True,
        cmap='coolwarm',
        fmt='.2f',
        center=0,
        square=True
    )

    plt.title('데이터 간 상관관계')
    plt.tight_layout()
    plt.savefig(base_path / f'figures/{fname}.svg')
    plt.close()

if __name__ == '__main__':
    df_0 = pd.read_csv('data/data.csv').drop(columns=['temp_min_anom', 'temp_max_anom'])
    df_0['date'] = pd.to_datetime(df_0['date'])

    df_30 = pd.read_csv('data/data_lag_30.csv').drop(columns=['temp_min_anom_30', 'temp_max_anom_30'])
    df_30['date'] = pd.to_datetime(df_30['date'])

    df_60 = pd.read_csv('data/data_lag_60.csv').drop(columns=['temp_min_anom_60', 'temp_max_anom_60'])
    df_60['date'] = pd.to_datetime(df_60['date'])

    df_90 = pd.read_csv('data/data_lag_90.csv').drop(columns=['temp_min_anom_90', 'temp_max_anom_90'])
    df_90['date'] = pd.to_datetime(df_90['date'])

    plot_scatter_all_data(df_0, 'scatter_0')
    plot_scatter_all_data(df_30, 'scatter_30')
    plot_scatter_all_data(df_60, 'scatter_60')
    plot_scatter_all_data(df_90, 'scatter_90')

    plot_heatmap_all_data(df_0, 'heatmap_0')
    plot_heatmap_all_data(df_30, 'heatmap_30')
    plot_heatmap_all_data(df_60, 'heatmap_60')
    plot_heatmap_all_data(df_90, 'heatmap_90')
    