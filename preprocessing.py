import pandas as pd
from pathlib import Path


BASE = Path(__file__).resolve().parent

def preprocess(base_path: Path = BASE):
    """데이터 전처리 함수"""
    print('===전처리 함수 동작 시작===')

    # csv 읽기
    climate = pd.read_csv(base_path / 'raw_data/climate.csv', encoding='utf-8-sig')
    price1 = pd.read_csv(base_path / 'raw_data/kimchi_cabbage_20210923.csv', encoding='utf-8-sig')
    price2 = pd.read_csv(base_path / 'raw_data/kimchi_cabbage_20230814.csv', encoding='utf-8-sig')
    price3 = pd.read_csv(base_path / 'raw_data/kimchi_cabbage_20231231.csv', encoding='utf-8-sig')
    price4 = pd.read_csv(base_path / 'raw_data/kimchi_cabbage_20241231.csv', encoding='utf-8-sig')
    price5 = pd.read_csv(base_path / 'raw_data/kimchi_cabbage_20260630.csv', encoding='utf-8-sig')
    normal = pd.concat(
        [pd.read_csv(base_path / f'raw_data/normal_climate_{m}.csv', skiprows=5, encoding='utf-8-sig') for m in range(1, 13)],
        ignore_index=True
    )
    print(normal.columns)

    # 데이터 통합
    price = pd.concat([price1, price2, price3, price4, price5])

    # 열 이름 변경
    climate = climate.rename(columns={    
        '지점': 'station',
        '지점명': 'region',
        '일시': 'date',
        '평균기온(°C)': 'temp_avg',
        '최저기온(°C)': 'temp_min',
        '최고기온(°C)': 'temp_max',
        '일강수량(mm)': 'rainfall',
        '평균 풍속(m/s)': 'wind_avg',
        '평균 상대습도(%)': 'humidity_avg',
        '합계 일조시간(hr)': 'sunshine_sum'
    })

    price = price.rename(columns={    
        '도매일자': 'date',
        '품목명': 'type',
        '도매일일가격': 'price'
    })

    normal = normal.rename(columns={
        '지점명': 'region',
        '지점번호': 'station',
        '일시': 'date',
        '평균기온(°C)': 'temp_avg_normal',
        '최저기온(°C)': 'temp_min_normal',
        '최고기온(°C)': 'temp_max_normal',
        '강수량(mm)': 'rainfall_normal',
        '평균풍속(m/sec)': 'wind_avg_normal',
        '습도(%)': 'humidity_avg_normal',
        '일조시간(hr)': 'sunshine_sum_normal'
    })

    # 필요한 열만 남기기
    vars_ = ['temp_avg', 'temp_min', 'temp_max', 'rainfall', 'wind_avg', 'humidity_avg', 'sunshine_sum']
    climate = climate[['station', 'region', 'date'] + vars_].copy()
    normal = normal[['station', 'date'] + [f'{v}_normal' for v in vars_]].copy()
    price = price[['date', 'type', 'price']].copy()

    # 날짜 처리
    climate['date'], price['date'] = pd.to_datetime(climate['date']), pd.to_datetime(price['date'])
    climate['month'], climate['day'] = climate['date'].dt.month, climate['date'].dt.day
    climate.loc[(climate['month'] == 2) & (climate['day'] == 29), 'day'] = 28
    md = normal['date'].astype(str).str.extract(r'(\d+)월\s*(\d+)일')
    print('날짜 형식이 아닌 일시 값:', normal.loc[md[0].isna(), 'date'].unique()[:10])
    normal[['month', 'day']] = md
    normal = normal.dropna(subset=['month', 'day'])
    normal[['month', 'day']] = normal[['month', 'day']].astype(int)
    normal_cols = [f'{v}_normal' for v in vars_]
    normal[normal_cols] = normal[normal_cols].apply(pd.to_numeric, errors='coerce')
    normal = normal.drop(columns='date')

    # 품목이 배추인 행만 남기기
    price = price[price['type'] == '배추'].drop(columns='type').reset_index(drop=True)

    # 비가 안 온 날은 강수량을 0으로 지정
    climate['rainfall'] = climate['rainfall'].fillna(0)

    # 관측소별로 평년값 붙이고 편차 계산
    print('normal 키 중복 수:', normal.duplicated(subset=['station', 'month', 'day']).sum())
    print('normal 결측치 수:', normal.isna().sum())
    normal = normal.drop_duplicates(subset=['station', 'month', 'day'])
    climate = climate.merge(normal, on=['station', 'month', 'day'], how='left')
    print(climate.loc[climate['temp_avg_normal'].isna(), 'region'].unique())
    print('평년값 안 붙은 행:', climate['temp_avg_normal'].isna().sum())
    for v in vars_:
        climate[f'{v}_anom'] = climate[v] - climate[f'{v}_normal']

    # 날짜별 전국 평균
    cols = [f'{v}_anom' for v in vars_]
    climate = climate.groupby('date', as_index=False)[cols].mean()

    # 중복값, 결측치 제거
    print('climate')
    print('중복값 수:', climate.duplicated().sum())
    print('결측치 수:', climate.isna().sum())
    climate = climate.drop_duplicates().reset_index(drop=True)
    climate = climate.dropna().reset_index(drop=True)

    print('price')
    print('중복값 수:', price.duplicated().sum())
    print('결측치 수:', price.isna().sum())
    price = price.drop_duplicates().reset_index(drop=True)
    price = price.dropna().reset_index(drop=True)

    # X 데이터와 y 데이터 통합
    df = pd.merge(price, climate, on='date', how='inner')

    # 시차 적용: 가격 날짜 기준 과거 데이터 편차 평균
    clim_daily = climate.set_index('date').sort_index().asfreq('D')  # 빠진 날짜를 채워서 1행 = 1일로 맞춤
    windows = {30: 1, 60: 31, 90: 61}  # 이름: 며칠 전부터 시작하는지

    for name, start in windows.items():
        lagged = clim_daily[cols].shift(start).rolling(30, min_periods=25).mean()
        lagged.columns = [f'{c}_{name}' for c in cols]
        lagged = lagged.reset_index()

        df_lag = pd.merge(price, lagged, on='date', how='inner').dropna().reset_index(drop=True)
        df_lag.to_csv(base_path / f'data/data_lag_{name}.csv', index=False, encoding='utf-8-sig')
        print(f'{name}일 윈도우 행 수:', len(df_lag))

    # 데이터 내보내기(csv)
    climate.to_csv(base_path / 'data/climate.csv', index=False, encoding='utf-8-sig')
    price.to_csv(base_path / 'data/price.csv', index=False, encoding='utf-8-sig')
    normal.to_csv(base_path / 'data/normal.csv', index=False, encoding='utf-8-sig')
    df.to_csv(base_path / 'data/data.csv', index=False, encoding='utf-8-sig')

    # 날짜 중복, 가격 결측치 확인
    print('날짜 중복 건수:', price.duplicated(subset='date').sum())
    print('가격 결측치(0) 건수:', (price['price'] == 0).sum())
    
    print('===전처리 함수 동작 끝===')

    return df


if __name__ == '__main__':
    df = preprocess()
    print(df.head())
    print(df.describe())
