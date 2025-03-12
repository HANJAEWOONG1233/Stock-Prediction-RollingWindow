import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
import talib 

# ------------- 사용자 설정 -------------
CSV_4H = 'NVDA_4H_DATA.csv'   # 4시간봉 원본 CSV 파일
WINDOW_SIZE = 100              # 롤링 학습 윈도우 크기
SCALING = True                 # 스케일링 적용 여부
OUTPUT_CSV = 'NVDA_prediction_mid_4h_only_strict_shifted.csv'

LGB_PARAMS = {
    'n_estimators': 100,
    'learning_rate': 0.05,
    'random_state': 42,
    'min_gain_to_split': np.float64(0.1),
    'min_data_in_leaf': 200,
    'num_leaves': 23,
    'max_depth': 5,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 9,
    'verbose': -1,
    'subsample': np.float64(0.0),
    'colsample_bytree': np.float64(0.0),
    'reg_alpha': np.float64(0.0),
    'reg_lambda': np.float64(0.1)
}
# --------------------------------------


def load_4h_data_with_more_indicators(csv_path) -> pd.DataFrame:
    # CSV를 읽을 때 parse_dates 대신 나중에 직접 변환합니다.
    df = pd.read_csv(csv_path)
    
    # time 컬럼이 Unix 타임스탬프(초 단위)인 경우, unit='s'를 지정합니다.
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    df.sort_index(inplace=True)

    df['high_low_medium'] = (df['high'] + df['low']) / 2

    # 이동평균
    df['SMA_10'] = talib.SMA(df['close'], timeperiod=10)
    df['SMA_20'] = talib.SMA(df['close'], timeperiod=20)
    df['SMA_50'] = talib.SMA(df['close'], timeperiod=50)

    df['EMA_10'] = talib.EMA(df['close'], timeperiod=10)
    df['EMA_20'] = talib.EMA(df['close'], timeperiod=20)
    df['EMA_50'] = talib.EMA(df['close'], timeperiod=50)

    # 볼린저 밴드
    df['BB_upper'], df['BB_middle'], df['BB_lower'] = talib.BBANDS(
        df['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
    )

    # TRIMA, WMA 
    df['TRIMA_30'] = talib.TRIMA(df['close'], timeperiod=30)
    df['WMA_30']   = talib.WMA(df['close'], timeperiod=30)

    # RSI, Stochastic, MACD 
    df['RSI_14'] = talib.RSI(df['close'], timeperiod=14)

    df['STOCH_k'], df['STOCH_d'] = talib.STOCH(
        df['high'], df['low'], df['close'],
        fastk_period=14, slowk_period=3, slowk_matype=0, 
        slowd_period=3, slowd_matype=0
    )

    df['STOCHRSI_k'], df['STOCHRSI_d'] = talib.STOCHRSI(
        df['close'],
        timeperiod=14,
        fastk_period=5,
        fastd_period=3,
        fastd_matype=0
    )

    df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(
        df['close'], fastperiod=12, slowperiod=26, signalperiod=9
    )

    df['WILLR'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=14)
    df['CCI']   = talib.CCI(df['high'], df['low'], df['close'], timeperiod=14)
    df['ROC']   = talib.ROC(df['close'], timeperiod=10)

    # OBV, AD, ADOSC, MFI
    df['OBV']   = talib.OBV(df['close'], df['Volume'])
    df['AD']    = talib.AD(df['high'], df['low'], df['close'], df['Volume'])
    df['ADOSC'] = talib.ADOSC(
        df['high'], df['low'], df['close'], df['Volume'],
        fastperiod=3, slowperiod=10
    )
    df['MFI']   = talib.MFI(
        df['high'], df['low'], df['close'], df['Volume'],
        timeperiod=14
    )

    # ATR, NATR, TRANGE
    df['ATR_14']  = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    df['NATR_14'] = talib.NATR(df['high'], df['low'], df['close'], timeperiod=14)
    df['TRANGE']  = talib.TRANGE(df['high'], df['low'], df['close'])

    # SAR
    df['sar_4h'] = talib.SAR(df['high'], df['low'], acceleration=0.011, maximum=0.2)

    # 결측값 NaN 처리
    df = df.fillna(method='ffill').fillna(method='bfill')

    return df


def rolling_window_prediction_and_export(df_4h: pd.DataFrame,
                                         window_size: int,
                                         scaling: bool,
                                         lgb_params: dict,
                                         output_csv: str) -> pd.DataFrame:
    df = df_4h.copy()

    # 1) 타겟: i번째 봉의 (high + low)/2
    df['target_mid_4h'] = (df['high'] + df['low']) / 2

    # 2) 지표 후보 컬럼들 (피처로 쓸 컬럼 확인) open, high, low, close, Volume, target_mid_4h는 제외
    exclude_cols = ['open', 'high', 'low', 'close', 'Volume', 'target_mid_4h']
    indicator_cols = [c for c in df.columns if c not in exclude_cols]

    # 3) 각 지표는 shift(1) 처리
    for col in indicator_cols:
        shifted_col = col + "_shifted"
        df[shifted_col] = df[col].shift(1)

    # 4) Feature 구성 : i번째 봉의 open + 모든 지표(shift된 것)
    feature_cols = ['open']  # i번째 봉의 open
    feature_cols += [col + "_shifted" for col in indicator_cols]

    # 5) 결측치 처리
    X = df[feature_cols].copy()
    y = df['target_mid_4h'].copy()

    X = X.fillna(method='ffill').fillna(method='bfill')
    y = y.fillna(method='ffill').fillna(method='bfill')

    # 6) 표준화 스케일링
    if scaling:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)

    data_len = len(df)
    predicted_mids = [np.nan] * data_len
    actual_mids = [np.nan] * data_len

    # 7) 롤링 윈도우 학습/예측
    for i in range(window_size, data_len):
        start_idx = i - window_size
        end_idx = i

        X_train = X.iloc[start_idx:end_idx]
        y_train = y.iloc[start_idx:end_idx]

        X_test = X.iloc[[i]]
        y_test = y.iloc[i]

        model = lgb.LGBMRegressor(**lgb_params)
        model.fit(X_train, y_train)

        pred_val = model.predict(X_test)[0]
        predicted_mids[i] = pred_val
        actual_mids[i] = y_test

    df['predicted_mid'] = predicted_mids
    df['actual_mid'] = actual_mids

    # 8) factor 적용 (롱/숏 구분)
    # factor가 0 ~ 0.3까지 0.001 단위로 반복
    factor_range = np.arange(0.0, 0.3001, 0.001)
    for factor in factor_range:
        col_name = f"pred_adj_{factor:.3f}"
        df[col_name] = np.where(
            df['predicted_mid'] > df['open'],
            df['predicted_mid'] * (1 - factor),
            df['predicted_mid'] * (1 + factor)
        )

    # 최종 저장 전 인덱스를 TIMESTAMP 형식 문자열로 변환 (예: 'YYYY-MM-DD HH:MM:SS')
    df.index = pd.to_datetime(df.index)
    df.index = df.index.strftime('%Y-%m-%d %H:%M:%S')
    
    df.to_csv(output_csv, index=True, encoding='utf-8')
    print(f"[INFO] CSV saved => {output_csv}")

    return df


def main():
    df_4h = load_4h_data_with_more_indicators(CSV_4H)
    print(f"[INFO] 4H Data shape: {df_4h.shape}")

    result_df = rolling_window_prediction_and_export(
        df_4h,
        window_size=WINDOW_SIZE,
        scaling=SCALING,
        lgb_params=LGB_PARAMS,
        output_csv=OUTPUT_CSV
    )

    print("[INFO] Done (middle value prediction + factor-adj)")


if __name__ == "__main__":
    main()
