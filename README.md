# Stock-Prediction-4-Hour Rolling Prediction & Factor-Based Backtesting


4시간봉 데이터를 이용해 봉의 중앙값을 예측하고, 예측 결과를 **factor**를 통해 조정 후 롱/숏 매매 시뮬레이션을 수행한 프로젝트입니다. 아래는 **코드 구조**와 **백테스트 결과**, 그리고 그래프를 삽입할 수 있도록 여유 공간을 둔 README 형식입니다.

---

## 1. Introduction

- **목표**  
  - (high + low)/2 형태의 봉 중앙값을 **LightGBM** 모델로 예측
  - **롱/숏** 매매 로직 적용 (예측값이 상승 → 롱, 하락 → 숏)
  - **factor**(0~0.3 범위, 0.001 단위)로 예측값 조정 후 **수익률**·**MDD** 등 성능 비교

- **특징**  
  - **Rolling Window** 방식: 일정 길이(window_size)의 과거 데이터를 사용해 직후 봉 예측
  - **TA-Lib** 활용: 이동평균, 볼린저 밴드, MACD 등 다양한 지표 자동 계산
  - **백테스트**: 초기자본 \$1,000, 고정 2주 매수/매도, 시가 진입·종가 청산, 수익률 및 MDD 분석

---

## 2. Code Structure Explanation

### A. Rolling Window Prediction
1. **데이터 불러오기**  
   - 4시간봉 CSV 파일 로드  
   - TA-Lib으로 여러 기술 지표 계산 (이동평균, 볼린저, MACD, RSI 등)

2. **특징(Feature) 구성**  
   - 예측 대상: \( (high + low)/2 \)  
   - 지표들은 i번째 봉 예측을 위해 **shift(1)** 처리 (과거 정보만 사용)

3. **LightGBM 모델 학습**  
   - **window_size**만큼의 과거 데이터를 슬라이딩 방식으로 학습  
   - i번째 시점 예측 후, 다시 (i+1)번째로 윈도우 이동

4. **factor 적용**  
   - 예측값이 open보다 크면 롱 → 예측값에 \((1 - factor)\) 곱  
   - 예측값이 open보다 작으면 숏 → 예측값에 \((1 + factor)\) 곱  
   - factor가 0~0.3까지 0.001 단위로 자동 반복

5. **결과 CSV 생성**  
   - 예측된 중간값 `predicted_mid`, factor 별 컬럼 `pred_adj_x.xx` 등 저장

---

### B. Factor-Based Backtesting
1. **백테스트 로직**  
   - 초기자본 \$1,000, 매 거래마다 2주를 매수/매도 (고정)  
   - 직전 봉 대비 예측값이 상승이면 **롱 진입**, 하락이면 **숏 진입**  
   - 시가(Open)에 진입, 종가(Close)에 청산

2. **수익률 및 에쿼티**  
   - 롱: \(\frac{\text{청산가} - \text{진입가}}{\text{진입가}}\), 숏: \(\frac{\text{진입가} - \text{청산가}}{\text{진입가}}\)  
   - 거래 후 이익(또는 손실)을 **현재 에쿼티**에 반영

3. **최대 드로우다운(MDD)**  
   - 백테스트 기간 중 에쿼티가 과거 최고점 대비 얼마나 하락했는지 측정

4. **factor별 비교**  
   - CSV에서 `pred_adj_0.000`부터 `pred_adj_0.300`까지 각각 백테스트  
   - **최고 수익률**의 factor를 찾아 에쿼티 곡선·드로우다운 그래프를 시각화

---

## 3. Hyperparameter Tuning

- 여러 후보 하이퍼파라미터(학습률, 트리 깊이 등)를 **Grid Search**로 탐색
- SMAPE(RMSE, MAE 가능)를 최소화하는 조합을 최종採用
- **경험적/실험적** 접근으로, 이론적 정답이라기보다는 최적 근사치

---

## 4. Final Backtest Results

아래 네 가지 자산(테슬라, 엔비디아, 애플, 비트코인)에 대해 **4시간봉** 데이터를 바탕으로 백테스트를 진행했습니다.  
**수익률**과 **MDD**를 간략히 정리합니다. 각 결과 아래에는 **그래프**(에쿼티/드로우다운)를 삽입할 공간을 배치했습니다.

---

### A. Tesla (4H, 2018.01 ~ 2025.01)

- **누적 수익률**: **214.54%** 
- **MDD**: **18.72%** 

![image](https://github.com/user-attachments/assets/466fdfe4-f5a1-4716-ab7e-abf6057c6cf5)



---

### B. Nvidia (4H, 2018.01 ~ 2025.01)

- **누적 수익률**: **84.83%** 
- **MDD**: **11.57%** 

![image](https://github.com/user-attachments/assets/f543e7b8-45e2-4bdc-bd24-7f5c7a317323)


---

### C. Apple (4H, 2018.01 ~ 2025.01)

- **누적 수익률**: **22.23%** 
- **MDD**: **3.11%** 

![image](https://github.com/user-attachments/assets/09909912-58d3-4943-ad2a-4945c082f7d4)


---

### D. Bitcoin (4H, 2022.07 ~ 2024.12)

- **누적 수익률**: **76.38%** 
- **MDD**: **15.29%** 

![image](https://github.com/user-attachments/assets/7f61b178-1416-4a11-a8fa-4ac31cc90aed)


---

## 5. Usage Guide

1. **데이터 준비**  
   - 4시간봉 CSV 파일 준비 (time, open, high, low, close, volume 필수)  
   - TA-Lib 사용 시 필요한 지표(예: RSI, MACD 등) 사전 설치

2. **Rolling 예측 실행**  
   - **롤링 윈도우**로 (고저 평균) 예측  
   - factor(0~0.3) 적용된 컬럼 생성

3. **백테스트 실행**  
   - factor별로 **수익률/드로우다운** 비교  
   - 최적 factor를 찾아 에쿼티 곡선, 드로우다운 그래프 확인

---

## 6. Notes & Future Work

- **수수료, 슬리피지** 추가 고려 → 실제 환경 접근성 강화
- **Leverage** 적용으로 전략 변동성·수익률 극대화 가능
- 4시간봉 이외의 **다양한 타임프레임**(1H, Daily)에서도 동일 로직 확장 가능

---

## 7. License

- 본 프로젝트의 모든 자료는 별도 명시가 없는 경우 [MIT License](https://opensource.org/licenses/MIT)를 따릅니다.

---

## 8. Final Summary

- **(고저 평균) 예측 + factor 조정**을 통한 **4시간봉 롱/숏 매매** 전략
- 테슬라, 엔비디아, 애플, 비트코인 모두 **양호한 누적 수익률** 달성
- 향후 **추가 지표**, **수수료** 반영, **레버리지** 조절 등으로 전략 고도화 가능





