import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===== 사용자 조정 변수 =====
COMMISSION_RATE = 0    # 거래 수수료 (예: 0.035% → 0.00035)
# (LEVERAGE는 사용하지 않으므로 제거 가능)
# ============================

def backtest_open_close_mid_single(df: pd.DataFrame, pm_col: str):
    """
    df 내 특정 컬럼(pm_col)을 '예측 중앙값' 또는 '조정된 예측값'이라고 가정하고,
    각 봉의 open, high, low, close, 그리고 pm_col 값에 근거해
    매매 시뮬레이션을 진행한 뒤 결과를 반환합니다.
    
    이번 버전은 초기 시드 $1,000에서 시작하여 매 거래마다 고정으로
    주식 2주를 매수(또는 매도)하는 방식으로 진입하며, 해당 시점의 주가를 반영하여
    거래 수익률, 누적 자본(에쿼티), MDD 등을 산출합니다.
    
    Parameters
    ----------
    df : pd.DataFrame
        'open', 'high', 'low', 'close', pm_col 등을 포함한 시계열 데이터
    pm_col : str
        예측값(또는 factor-adjusted 예측값) 컬럼명
    
    Returns
    -------
    tuple(pd.DataFrame, dict)
        - 백테스트가 적용된 DataFrame (trade_return, equity, drawdown, profit 컬럼 포함)
        - 성능 요약 딕셔너리(평균 수익률, 누적 수익률, MDD 등)
    """
    temp_df = df.copy()

    # 필수 컬럼 체크
    required_cols = ['open', 'high', 'low', 'close', pm_col]
    for c in required_cols:
        if c not in temp_df.columns:
            raise ValueError(f"Required column '{c}' not found in DataFrame.")

    # 초기 시드 및 자본 변수
    INITIAL_SEED = 1000.0
    current_equity = INITIAL_SEED

    # 매 거래마다 고정으로 매수(또는 매도)할 주식 수
    SHARE_QUANTITY = 2

    # 결과 저장용 컬럼 초기화
    temp_df['trade_return'] = 0.0   # 거래별 수익률 (%)
    temp_df['profit'] = 0.0         # 거래별 달러 수익
    temp_df['position'] = None      # 포지션 상태
    temp_df['equity'] = np.nan      # 각 봉의 누적 자본 (에쿼티)

    # 첫 행의 에쿼티는 초기 자본으로 설정
    temp_df.at[temp_df.index[0], 'equity'] = current_equity

    # 초기 포지션 상태
    current_position = None
    entry_price = None  # 거래 진입 시 주당 가격 (수수료 포함)

    # 매매 시뮬레이션 (한 봉 단위)
    for i in range(1, len(temp_df)):
        # 이전 행의 에쿼티 기록
        temp_df.at[temp_df.index[i-1], 'equity'] = current_equity

        prev_pm = temp_df[pm_col].iloc[i-1]
        curr_pm = temp_df[pm_col].iloc[i]

        o = temp_df['open'].iloc[i]
        c = temp_df['close'].iloc[i]

        # 포지션이 없으면 진입 신호 확인
        if current_position is None:
            if curr_pm > prev_pm:
                # 롱 포지션 진입: 주식 2주 매수 (매수 시 수수료 반영)
                current_position = 'long'
                entry_price = o * (1 + COMMISSION_RATE)
                temp_df.at[temp_df.index[i], 'position'] = 'long_enter'
            elif curr_pm < prev_pm:
                # 숏 포지션 진입: 주식 2주 매도 (매도 시 수수료 반영)
                current_position = 'short'
                entry_price = o * (1 - COMMISSION_RATE)
                temp_df.at[temp_df.index[i], 'position'] = 'short_enter'
            else:
                temp_df.at[temp_df.index[i], 'position'] = None

        # 포지션 보유 중 → 청산 신호 확인
        else:
            if current_position == 'long':
                if curr_pm < prev_pm:
                    # 롱 포지션 청산: 주식 2주 매도 (매도 시 수수료 반영)
                    exit_price = c * (1 - COMMISSION_RATE)
                    pct_return = (exit_price - entry_price) / entry_price
                    # 거래 비용 = entry_price * SHARE_QUANTITY
                    profit = SHARE_QUANTITY * (exit_price - entry_price)
                    current_equity += profit
                    temp_df.at[temp_df.index[i], 'trade_return'] = pct_return
                    temp_df.at[temp_df.index[i], 'profit'] = profit
                    current_position = None
                    temp_df.at[temp_df.index[i], 'position'] = 'long_exit'
                else:
                    temp_df.at[temp_df.index[i], 'position'] = 'long_hold'
            elif current_position == 'short':
                if curr_pm > prev_pm:
                    # 숏 포지션 청산: 주식 2주 매수 (매수 시 수수료 반영)
                    exit_price = c * (1 + COMMISSION_RATE)
                    pct_return = (entry_price - exit_price) / entry_price
                    profit = SHARE_QUANTITY * (entry_price - exit_price)
                    current_equity += profit
                    temp_df.at[temp_df.index[i], 'trade_return'] = pct_return
                    temp_df.at[temp_df.index[i], 'profit'] = profit
                    current_position = None
                    temp_df.at[temp_df.index[i], 'position'] = 'short_exit'
                else:
                    temp_df.at[temp_df.index[i], 'position'] = 'short_hold'

    # 만약 마지막 봉에서 포지션이 남아있다면 청산
    if current_position is not None:
        last_index = temp_df.index[-1]
        last_close = temp_df['close'].iloc[-1]
        if current_position == 'long':
            exit_price = last_close * (1 - COMMISSION_RATE)
            pct_return = (exit_price - entry_price) / entry_price
            profit = SHARE_QUANTITY * (exit_price - entry_price)
            current_equity += profit
            temp_df.at[last_index, 'trade_return'] = pct_return
            temp_df.at[last_index, 'profit'] = profit
            temp_df.at[last_index, 'position'] = 'long_exit'
        elif current_position == 'short':
            exit_price = last_close * (1 + COMMISSION_RATE)
            pct_return = (entry_price - exit_price) / entry_price
            profit = SHARE_QUANTITY * (entry_price - exit_price)
            current_equity += profit
            temp_df.at[last_index, 'trade_return'] = pct_return
            temp_df.at[last_index, 'profit'] = profit
            temp_df.at[last_index, 'position'] = 'short_exit'
    # 마지막 행 에쿼티 업데이트
    temp_df.at[temp_df.index[-1], 'equity'] = current_equity

    # 누적 에쿼티 채우기
    temp_df['equity'].ffill(inplace=True)
    
    # 드로우다운 계산 (누적 최대치 대비 하락률)
    rolling_max = temp_df['equity'].cummax()
    temp_df['drawdown'] = (rolling_max - temp_df['equity']) / rolling_max
    mdd_pct = temp_df['drawdown'].max() * 100.0

    # 최종 누적 수익률 (%)
    total_return_pct = ((current_equity - INITIAL_SEED) / INITIAL_SEED) * 100.0

    # 거래별 수익률(진입-청산 시 pct_return) 평균 (백분율)
    trade_returns = temp_df.loc[temp_df['trade_return'] != 0, 'trade_return']
    if not trade_returns.empty:
        avg_return_pct = trade_returns.mean() * 100.0
    else:
        avg_return_pct = 0.0

    # 결과 요약
    result = {
        'pm_col': pm_col,
        'avg_return_%': avg_return_pct,
        'total_return_%': total_return_pct,
        'mdd_%': mdd_pct,
        'shares': SHARE_QUANTITY
    }

    return temp_df, result

def backtest_multiple_factors(csv_file: str):
    """
    1. csv_file 로드
    2. pred_adj_로 시작하는 factor 컬럼들을 자동 탐색
    3. 각 factor 컬럼에 대해 backtest_open_close_mid_single() 실행
    4. 결과(평균 수익률/누적 수익률/MDD)를 DataFrame으로 모아서
       정렬/출력 후 반환
    """
    df = pd.read_csv(csv_file, parse_dates=['time'], index_col='time')

    # pred_adj_로 시작하는 factor 컬럼 탐색
    factor_cols = [col for col in df.columns if col.startswith("pred_adj_")]
    
    results_list = []
    best_bt_df = None   # 최적 factor에 해당하는 DataFrame 저장용 변수
    best_total_return = -np.inf

    # 각 factor 컬럼별 백테스트 실행
    for factor_col in factor_cols:
        bt_df, bt_result = backtest_open_close_mid_single(df, pm_col=factor_col)
        results_list.append(bt_result)
        if bt_result['total_return_%'] > best_total_return:
            best_total_return = bt_result['total_return_%']
            best_bt_df = bt_df

    # 결과 테이블 생성
    results_df = pd.DataFrame(results_list)

    if results_df.empty:
        print("[WARN] No factor columns (pred_adj_*) found in CSV.")
        return results_df, None

    # pm_col 내 숫자 부분 추출 후 float 변환하여 정렬
    results_df['pm_numeric'] = results_df['pm_col'].str.extract(r'(\d+\.\d+)')[0].astype(float)
    results_df.sort_values('pm_numeric', inplace=True)
    results_df.drop(columns=['pm_numeric'], inplace=True)
    results_df.reset_index(drop=True, inplace=True)

    # 콘솔 출력
    print("\n=== Factor별 백테스트 결과 ===")
    print(results_df)

    best_factor_idx = results_df['total_return_%'].idxmax()
    best_factor_row = results_df.loc[best_factor_idx]
    print(f"\n[INFO] Best Factor by total_return: {best_factor_row['pm_col']}")
    print(best_factor_row)

    return results_df, best_bt_df

if __name__ == "__main__":
    csv_file_path = "NVDA_prediction_mid_4h_only_strict_shifted.csv"
    results, best_df = backtest_multiple_factors(csv_file_path)

    if best_df is not None:
        best_factor = results.loc[results['total_return_%'].idxmax(), 'pm_col']
        
        plt.figure(figsize=(12, 8))
        
        # 에쿼티 곡선
        plt.subplot(2, 1, 1)
        plt.plot(best_df.index, best_df['equity'], label='Equity', color='blue')
        plt.title(f"Equity Curve for Best Factor: {best_factor} (Fixed 2 Shares per Trade)")
        plt.ylabel("Equity")
        plt.legend()
        plt.grid(True)
        
        # 드로우다운 곡선
        plt.subplot(2, 1, 2)
        plt.plot(best_df.index, best_df['drawdown'], label='Drawdown', color='red')
        plt.title(f"Drawdown for Best Factor: {best_factor}")
        plt.xlabel("Time")
        plt.ylabel("Drawdown")
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

