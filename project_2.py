import pandas as pd
import joblib
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 데이터 준비
df = pd.read_csv("updated_data_with_week(2).csv")  # 데이터 불러오기

# Actual_Arrival_Time과 Estimated_Arrival_Time 컬럼을 datetime 형식으로 변환
df["Actual_Arrival_Time"] = pd.to_datetime(df["Actual_Arrival_Time"])
df["Estimated_Arrival_Time"] = pd.to_datetime(df["Estimated_Arrival_Time"])

# Delay_Hours 계산 (예상 도착 시간 - 실제 도착 시간)
df["Delay_Hours"] = (df["Estimated_Arrival_Time"] - df["Actual_Arrival_Time"]).dt.total_seconds() / 3600  # 초 -> 시간

# 사용하려는 피처와 타겟 정의
features = ["Inventory_Level", "Temperature", "Humidity", "Waiting_Time", 
            "User_Transaction_Amount", "User_Purchase_Frequency", "Asset_Utilization", 
            "Fuel_Cost", "Shipment_Status", "Traffic_Status", "Weather_Condition", "Road_Type"]
X = pd.get_dummies(df[features], drop_first=True)
y = df["Delay_Hours"]

# # 데이터 정규화
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # 지연 시간(y) 정규화
# y_scaler = StandardScaler()
# y_scaled = y_scaler.fit_transform(y.values.reshape(-1, 1))

# # 학습 데이터 분리
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# # 선형 회귀 모델 학습
# model = LinearRegression()
# model.fit(X_train, y_train)

# # 모델 저장
# joblib.dump(model, "linear_regression_model.pkl")
# joblib.dump(scaler, "scaler.pkl")
# joblib.dump(y_scaler, "y_scaler.pkl")

# print("✅ 모델, 스케일러 저장 완료!")

# ---------------------- #
# 모델 및 스케일러 로드 #
# ---------------------- #

# 모델 및 스케일러 로드
@st.cache_resource
def load_model():
    model = joblib.load("linear_regression_model.pkl")  # 저장된 모델 불러오기
    scaler = joblib.load("scaler.pkl")  # 저장된 스케일러 불러오기
    y_scaler = joblib.load("y_scaler.pkl")  # 저장된 y_scaler 불러오기
    return model, scaler, y_scaler

model, scaler, y_scaler = load_model()

# Streamlit UI 구성
st.title("📦 이차전지 물류 지연 시간 예측")
st.write("특정 조건을 입력하면 예상 지연 시간을 예측합니다.")

# 사용자 입력 받기
temperature = st.number_input("🌡️ 온도 (Temperature)", min_value=-10, max_value=50, value=25)
humidity = st.number_input("💧 습도 (Humidity)", min_value=0, max_value=100, value=60)
traffic_status = st.selectbox("🚦 교통 상황 (Traffic Status)", ["Light", "Moderate", "Heavy"])
weather_condition = st.selectbox("🌦️ 날씨 상태 (Weather Condition)", ["Clear", "Rainy", "Snowy"])

# 나머지 특성들에 대한 기본값 설정
inventory_level = 300
waiting_time = 14
transaction_amount = 300
purchase_frequency = 5
asset_utilization = 0.8
fuel_cost = 100
shipment_status = "Delivered"  # "Pending", "In Transit", "Delivered" 중 하나
road_type = "Highway"  # "Highway", "City Road", "Rural Road" 중 하나

# 입력 데이터 처리
input_data = pd.DataFrame({
    "Temperature": [temperature],
    "Humidity": [humidity],
    "Traffic_Status": [traffic_status],
    "Weather_Condition": [weather_condition],
    "Inventory_Level": [inventory_level],
    "Waiting_Time": [waiting_time],
    "User_Transaction_Amount": [transaction_amount],
    "User_Purchase_Frequency": [purchase_frequency],
    "Asset_Utilization": [asset_utilization],
    "Fuel_Cost": [fuel_cost],
    "Shipment_Status": [shipment_status],
    "Road_Type": [road_type]
})

# 범주형 변수 원-핫 인코딩
input_data_encoded = pd.get_dummies(input_data, drop_first=True)

# 기존 데이터와 동일한 컬럼 맞추기
missing_cols = set(X.columns) - set(input_data_encoded.columns)
for col in missing_cols:
    input_data_encoded[col] = 0

# 컬럼 순서 정렬 (X의 컬럼 순서와 맞추기)
input_data_encoded = input_data_encoded[X.columns]

# 데이터 정규화
input_data_scaled = scaler.transform(input_data_encoded)

# 예측 및 결과 출력
if st.button("🚀 지연 시간 예측하기"):
    predicted_delay_scaled = model.predict(input_data_scaled)
    
    # 예측된 값 역변환 (원래 단위로 복원)
    predicted_delay = y_scaler.inverse_transform(predicted_delay_scaled.reshape(-1, 1))

    # 예측된 지연 시간 출력 (분으로 변환)
    predicted_delay_minutes = predicted_delay[0][0] * 60  # 시간 -> 분으로 변환
    st.success(f"⏳ 예상 지연 시간: {predicted_delay_minutes:.2f} 분")
