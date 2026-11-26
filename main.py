import warnings
warnings.filterwarnings(action='ignore')

import pandas as pd
import koreanize_matplotlib
import matplotlib.pyplot as plt

from pathlib import Path

import preprocessing
import analysis as anal
import modeling


BASE = Path(__file__).resolve().parent

def main():
    # 전처리
    print('===전처리===')
    try:
        df = pd.read_csv(BASE / 'data/data_lag_30.csv')
    except FileNotFoundError:
        preprocessing.preprocess()
        df = pd.read_csv(BASE / 'data/data_lag_30.csv')

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
    print('===모델 학습 및 평가===')
    combos = {
        'all': ['temp_avg_anom_30', 'wind_avg_anom_30', 'humidity_avg_anom_30', 'sunshine_sum_anom_30'],
        'top3': ['temp_avg_anom_30', 'wind_avg_anom_30', 'humidity_avg_anom_30'],
        'top2': ['temp_avg_anom_30', 'wind_avg_anom_30'],
        'top1': ['temp_avg_anom_30'],
    }

    df = pd.read_csv('data/data_lag_30.csv', parse_dates=['date'])
    df = df.drop(columns=['temp_min_anom_30', 'temp_max_anom_30', 'rainfall_anom_30']).sort_values('date')

    X_train, X_test, y_train, y_test = split_data(df)

    kimchi_predicted = {}
    for name, cols in combos.items():
        model = modeling.train(X_train[cols], y_train)
        predicted = modeling.evaluate(model, f'actual_predicted_{name}.svg', X_test[cols], y_test)
        predicted = pd.Series(predicted, index=X_test.index)   # 날짜와 짝 맞추려고 인덱스 붙임
        kimchi_predicted[name] = calculate_kimchi_price(predicted)
    print('===모델 학습 및 평가 끝===')

    # 김치 가격 도출
    print('===김치 가격 도출===')
    test_dates = df.loc[X_test.index, 'date']              # 테스트 구간 날짜
    kimchi_actual = calculate_kimchi_price(y_test)         # 실제 배추 가격으로 계산한 김치 가격

    plt.figure(figsize=(16, 6))
    plt.plot(test_dates, kimchi_actual, color='black', linewidth=2, label='실제 배추 가격으로 계산한 김치 가격')
    for name, kimchi in kimchi_predicted.items():
        plt.plot(test_dates, kimchi, alpha=0.8, label=f'예측 김치 가격({name})')

    plt.xlabel('날짜')
    plt.ylabel('김치 가격(원)')
    plt.title('실제 배추 가격으로 계산한 김치 가격 vs. 예측 김치 가격')
    plt.legend()
    plt.tight_layout()
    plt.savefig(BASE / 'figures/kimchi_price.svg')
    plt.close()

    # 결과 표로 저장
    result = pd.DataFrame({'date': test_dates, 'kimchi_actual': kimchi_actual,
                           **{f'kimchi_{n}': v for n, v in kimchi_predicted.items()}})
    result.to_csv(BASE / 'data/kimchi_price.csv', index=False, encoding='utf-8-sig')
    print('===김치 가격 도출 끝===')

    # 기후 변화 시나리오
    combo_labels = {'all': '기온·풍속·습도·일조량', 'top3': '기온·풍속·습도', 'top2': '기온·풍속', 'top1': '기온'}
    scenario_df = simulate_temp_scenario(X_train, y_train, X_test, combos, combo_labels)
    plot_temp_scenario(scenario_df, 'kimchi_scenario')

    print('===시나리오별 평균 김치 가격===')
    print(scenario_df.round(0))
    print('===현재 대비 상승률(%)===')
    print((scenario_df / scenario_df.loc['현재'] - 1).mul(100).round(2))
    scenario_df.to_csv(BASE / 'data/kimchi_scenario.csv', encoding='utf-8-sig')


def split_data(data: pd.DataFrame):
    # 날짜 기준으로 앞부분은 학습, 뒷부분은 평가
    split = int(len(data) * 0.8)
    train_data, test_data = data.iloc[:split], data.iloc[split:]

    X_train, y_train = train_data.drop(columns=['date', 'price']), train_data['price']
    X_test, y_test = test_data.drop(columns=['date', 'price']), test_data['price']

    return X_train, X_test, y_train, y_test


def calculate_kimchi_price(cabbage_price: pd.Series):
    return (cabbage_price + 6500) / 2.25


def simulate_temp_scenario(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame,
                           combos: dict, combo_labels: dict, scenarios: list = [0, 1, 2, 3]):
    """기온 편차를 올렸을 때 조합별 평균 김치 가격 계산"""
    scenario_result = {}
    for name, cols in combos.items():
        model = modeling.train(X_train[cols], y_train)
        means = []
        for delta in scenarios:
            X_scn = X_test[cols].copy()
            X_scn['temp_avg_anom_30'] += delta   # 기온만 올리고 나머지 변수는 그대로
            cabbage = pd.Series(model.predict(X_scn), index=X_scn.index)
            means.append(calculate_kimchi_price(cabbage).mean())
        scenario_result[combo_labels[name]] = means

    return pd.DataFrame(scenario_result, index=['현재' if d == 0 else f'+{d}℃' for d in scenarios])


def plot_temp_scenario(scenario_df: pd.DataFrame, fname: str, base_path: Path = BASE):
    """시나리오별 현재 대비 김치 가격 변화 막대그래프"""
    increase_df = scenario_df.sub(scenario_df.loc['현재']).drop(index='현재')

    ax = increase_df.plot(kind='bar', figsize=(10, 6), rot=0)
    for container in ax.containers:
        ax.bar_label(container, fmt='%+.0f', fontsize=9)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xlabel('기온 편차 증가량')
    ax.set_ylabel('현재 대비 김치 가격 변화(원)')
    ax.set_title('기온 상승 시나리오별 김치 가격 변화 예상')
    ax.legend(title='사용한 변수')
    plt.tight_layout()
    plt.savefig(base_path / f'figures/{fname}.svg')
    plt.close()


if __name__ == '__main__':
    main()
