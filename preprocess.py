import pandas as pd
from pathlib import Path


BASE = Path(__file__).resolve().parent

def preprocess(base_path):
    """데이터 전처리 함수"""
    print('===전처리 함수 동작 시작===')

    # csv 읽기
    climate = pd.read_csv(base_path / 'raw_data/climate.csv', encoding='utf-8-sig')
    price1 = pd.read_csv(base_path / 'raw_data/kimchi_cabbage_20210923.csv', encoding='utf-8-sig')
    price2 = pd.read_csv(base_path / 'raw_data/kimchi_cabbage_20230814.csv', encoding='utf-8-sig')
    price3 = pd.read_csv(base_path / 'raw_data/kimchi_cabbage_20231231.csv', encoding='utf-8-sig')
    price4 = pd.read_csv(base_path / 'raw_data/kimchi_cabbage_20241231.csv', encoding='utf-8-sig')
    price5 = pd.read_csv(base_path / 'raw_data/kimchi_cabbage_20260630.csv', encoding='utf-8-sig')

    # 데이터 통합
    price = pd.concat([price1, price2, price3, price4, price5])

    # 열 이름 변경
    climate = climate.rename(columns={    
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

    # 필요한 열만 남기기
    climate = climate[['region', 'date', 'temp_avg', 'temp_min', 'temp_max', 'rainfall', 'wind_avg', 'humidity_avg', 'sunshine_sum']]
    price = price[['date', 'type', 'price']]

    # 날짜 형 변환
    climate['date'], price['date'] = pd.to_datetime(climate['date']), pd.to_datetime(price['date'])

    # 품목이 배추인 행만 남기기
    price = price[price['type'] == '배추'].reset_index(drop=True)
    price = price.drop(columns='type')

    # 비가 안 온 날은 강수량을 0으로 지정
    climate['rainfall'] = climate['rainfall'].fillna(0)

    # 지역별로 나뉜 기후 데이터를 이용해 날짜별 전국 평균 구하기
    cols = ['temp_avg', 'temp_min', 'temp_max', 'rainfall', 'wind_avg', 'humidity_avg', 'sunshine_sum']
    climate = climate.groupby('date', as_index=False)[cols].mean()
    

    # 중복값, 결측치 제거
    print('중복값 수:', climate.duplicated().sum())
    print('결측치 수:', climate.isna().sum())
    climate = climate.drop_duplicates().reset_index(drop=True)
    climate = climate.dropna().reset_index(drop=True)

    print('중복값 수:', price.duplicated().sum())
    print('결측치 수:', price.isna().sum())
    price = price.drop_duplicates().reset_index(drop=True)
    price = price.dropna().reset_index(drop=True)

    # X 데이터와 y 데이터 통합
    df = pd.merge(price, climate, on='date', how='inner')

    # 데이터 내보내기(csv)
    climate.to_csv(base_path / 'data/climate.csv', index=False, encoding='utf-8-sig')
    price.to_csv(base_path / 'data/price.csv', index=False, encoding='utf-8-sig')
    df.to_csv(base_path / 'data/data.csv', index=False, encoding='utf-8-sig')

    # 날짜 중복, 가격 결측치 확인
    print('날짜 중복 건수:', price.duplicated(subset='date').sum())
    print('가격 결측치(0) 건수:', (price['price'] == 0).sum())
    
    print('===전처리 함수 동작 끝===')

    return df


if __name__ == '__main__':
    df = preprocess(BASE)
    print(df.head())
