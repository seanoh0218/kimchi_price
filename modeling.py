import warnings
warnings.filterwarnings(action='ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import koreanize_matplotlib

from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


BASE = Path(__file__).resolve().parent

def train(X: pd.DataFrame, y: pd.Series):
    model = LinearRegression()
    model.fit(X, y)

    print(model.coef_, model.intercept_)    
    return model


def evaluate(model, fname: str, X_test: pd.DataFrame, y_test: pd.Series, base_path: Path = BASE):
    # 검증 데이터의 배추 가격 예측하기   # ★ 주석 수정 (체지방률 → 배추 가격)
    y_pred = model.predict(X_test)

    # R2와 RMSE를 이용해 모델 평가하기
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print('모든 기상 관측값 사용')
    print(f'R2 = {r2:.3f}')
    print(f'RMSE = {rmse:.2f}')

    fig, ax = plt.subplots()   # ★ ax를 받아야 박스 위치를 고정할 수 있음
    ax.scatter(y_test, y_pred, alpha=0.7)

    min_v = min(y_test.min(), y_pred.min())
    max_v = max(y_test.max(), y_pred.max())
    ax.plot([min_v, max_v], [min_v, max_v], linestyle='--')

    # ★ RMSE, R2 박스
    ax.text(
        0.03, 0.97,
        f'RMSE = {rmse:,.0f}원\n$R^2$ = {r2:.3f}',
        transform=ax.transAxes,
        va='top', ha='left', fontsize=11,
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9)
    )

    ax.set_xlabel('실제 배추 가격')
    ax.set_ylabel('예측 배추 가격')
    ax.set_title('실제 배추 가격 vs. 예측 배추 가격')
    plt.tight_layout()   # ★ 라벨 잘림 방지
    plt.savefig(base_path / 'figures/actual_predicted.svg')
    plt.close()


if __name__ == '__main__':
    df = pd.read_csv('data/data_lag_30.csv', parse_dates=['date'])
    df = df.drop(columns=['temp_min_anom_30', 'temp_max_anom_30']).sort_values('date')

    # 날짜 기준으로 앞부분은 학습, 뒷부분은 평가
    split = int(len(df) * 0.8)
    train_data, test_data = df.iloc[:split], df.iloc[split:]

    X_train, y_train = train_data.drop(columns=['date', 'price']), train_data['price']
    X_test, y_test = test_data.drop(columns=['date', 'price']), test_data['price']

    model = train(X_train, y_train)
    evaluate(model, 'actual_predict', X_test, y_test)
