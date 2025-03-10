import streamlit as st
import pandas as pd
import pydeck as pdk
import plotly.express as px
import re
from geopy.distance import geodesic
import plotly.graph_objects as go

# ✅ 전체 페이지 너비 확장
st.set_page_config(layout="wide")

# 📌 CSV 데이터 불러오기
@st.cache_data
def load_data():
    df = pd.read_csv("updated_data_with_week_최종.csv")  # 🔹 CSV 파일 경로 설정
    return df

# 📌 Timestamp 관련 데이터 정리 함수
def clean_timestamp_columns(df, columns):
    for col in columns:
        df[col] = df[col].apply(lambda x: re.findall(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}', str(x)))
        df[col] = df[col].apply(lambda x: x[0] if isinstance(x, list) and x else None)
        df[col] = pd.to_datetime(df[col], errors="coerce")

        if df[col].isnull().any():
            st.warning(f"⚠️ 일부 {col} 값이 잘못된 형식으로 처리되었습니다. NaT 값이 포함된 행이 있습니다.")

    return df

# ✅ 데이터 불러오기 및 정리
df = load_data()
df = clean_timestamp_columns(df, ["Timestamp", "Estimated_Arrival_Time", "Actual_Arrival_Time", "Departure_Time"])

# ✅ Delay_Time 계산 (Actual - Estimated)
df["Delay_Time"] = (df["Actual_Arrival_Time"] - df["Estimated_Arrival_Time"]).dt.total_seconds() / 60  # 분 단위 변환

# ✅ 📌 필터를 화면 최상단에 배치 (더 작게)
placeholder = st.empty()  # 🔹 필터를 넣을 자리 확보

with placeholder.container():
    col1, col2, col3 = st.columns([0.5, 1, 2.5])  # 첫 번째 칼럼을 작게 설정
    
    with col1:  # 🔹 작은 칼럼에 필터 배치 → 크기 축소
        selected_week = st.selectbox("", df["Week_Label"].unique(), label_visibility="collapsed")

# ✅ 🚛 대시보드 제목 (가운데 정렬)
st.markdown('<h1 style="text-align: center;">🚛🔋 이차전지 물류 대시보드 🔋🚛</h1>', unsafe_allow_html=True)

# ✅ 선택된 데이터 필터링
filtered_df = df[df["Week_Label"] == selected_week]

st.markdown(
    f"""
    <h3 style="text-align: right; font-size: 25px;">📦{selected_week} 물류 데이터📦</h3>
    """,
    unsafe_allow_html=True
)



## ✅ KPI 지표 섹션
with st.container():
    st.markdown(
        """
        <style>
        .metric-card {
            background-color: #f9f9f9;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
        }
        .metric-title {
            font-size: 16px;
            font-weight: bold;
            color: #333;
        }
        .metric-value {
            font-size: 22px;
            font-weight: bold;
            color: #007bff;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3, col4, col5 = st.columns(5)
    
    # 색상 기준 설정 함수
    def get_color(value, threshold_low, threshold_high):
        if value < threshold_low or value > threshold_high:
            return 'red'
        return '#007bff'

    with col1:
        # 🚛 이미 필터링된 데이터 (주 단위) 사용
        operated_vehicles = filtered_df["Asset_ID"].nunique()  # 특정 기간 내 운행된 차량 수
        total_vehicles = df["Asset_ID"].nunique()  # 전체 차량 수

        # 🚛 차량 가동률 계산 (백분율)
        vehicle_utilization = (operated_vehicles / total_vehicles) * 100 if total_vehicles > 0 else 0

        # 🚛 Streamlit UI에 차량 가동률 표시
        st.markdown(
            f"""
            <div class="metric-card">
                🚛<div class="metric-title">차량 가동률(%)</div>
                <div class="metric-value">{vehicle_utilization:.1f}%</div>
            </div>
            """, 
            unsafe_allow_html=True
        )

    with col2:
        delay_rate = (filtered_df['Logistics_Delay'].mean() * 100)
        st.markdown(f'<div class="metric-card">⚠️<div class="metric-title">지연 발생률(%)</div>'
                f'<div class="metric-value" style="color: {get_color(delay_rate, 0, 30)};">{delay_rate:.1f}%</div></div>',
                unsafe_allow_html=True)

    with col3:
        avg_delay_time = filtered_df['Delay_Time'].mean()
        st.markdown(f'<div class="metric-card">🕒<div class="metric-title">평균 지연시간(분)</div>'
                f'<div class="metric-value" style="color: {get_color(avg_delay_time, 0, 20)};">{avg_delay_time:.1f} 분</div></div>',
                unsafe_allow_html=True)
    with col4:
        temp_humidity_compliance = ((filtered_df['Temperature'].between(0, 25) & 
                                 filtered_df['Humidity'].between(0, 55)).mean() * 100)
        st.markdown(f'<div class="metric-card">🌡<div class="metric-title">적정 온습도 준수율(%)</div>'
                f'<div class="metric-value" style="color: {get_color(temp_humidity_compliance, 50, 100)};">'
                f'{temp_humidity_compliance:.1f}%</div></div>',
                unsafe_allow_html=True)
        
    with col5:
        accuracy_threshold = 10  
        filtered_df["ETA_Accuracy"] = (filtered_df["Delay_Time"].abs() <= accuracy_threshold)
        eta_accuracy = filtered_df["ETA_Accuracy"].mean() * 100
        
        st.markdown('<div class="metric-card">📅<div class="metric-title">ETA 정확도(%)</div><div class="metric-value">' +
                    f"{eta_accuracy:.1f}%</div></div>", unsafe_allow_html=True)


# ✅ 제목 스타일 정의 (테두리 추가)
st.markdown(
    """
    <style>
    .title-box {
        background-color: #f9f9f9;
        padding: 10px;
        border-radius: 8px;
        text-align: center;
        font-size: 18px;
        font-weight: bold;
        color: #333;
        margin-bottom: 10px;
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 📌 지도 시각화 & 호차별 운행 현황 (2개를 가로로 배치)
with st.container():
    col1, col2 = st.columns([2, 1])  # ✅ 지도는 크게, 운행 현황은 작게
    with col1:
        st.markdown('<div class="title-box">🗺 물류 경로 시각화</h2>', unsafe_allow_html=True)
        if filtered_df.empty:
            st.warning("선택한 주에 대한 데이터가 없습니다.")
        else:
            arc_layer = pdk.Layer(
                "ArcLayer",
                data=filtered_df,
                get_source_position=["Origin_Longitude", "Origin_Latitude"],
                get_target_position=["Destination_Longitude", "Destination_Latitude"],
                get_width=3,
                get_source_color=[0, 128, 255, 100],
                get_target_color=[255, 0, 0, 100],
                auto_highlight=True,
                pickable=True,
            )
            view_state = pdk.ViewState(
                latitude=filtered_df["Origin_Latitude"].mean(),
                longitude=filtered_df["Origin_Longitude"].mean(),
                zoom=6,
                pitch=50,
            )
            st.pydeck_chart(pdk.Deck(map_style="mapbox://styles/mapbox/light-v9", initial_view_state=view_state, layers=[arc_layer]))


    # ✅ 호차별 운행 현황 (표 형태)
with col2:  
    st.markdown('<div class="title-box">🚚 호차별 운행 현황</h2>', unsafe_allow_html=True)

    # 🔹 트럭별 배송 횟수 계산
    shipment_count = filtered_df.groupby("Asset_ID")["Shipment_Status"].count().reset_index()
    shipment_count.columns = ["Asset_ID", "Shipment_Count"]

    # ✅ Haversine 공식을 사용한 총 이동 거리 계산
    def calculate_distance(row):
        start = (row["Origin_Latitude"], row["Origin_Longitude"])
        end = (row["Destination_Latitude"], row["Destination_Longitude"])
        return geodesic(start, end).kilometers  # km 단위 반환

    filtered_df["Distance"] = filtered_df.apply(calculate_distance, axis=1)
    total_distance = filtered_df.groupby("Asset_ID")["Distance"].sum().reset_index()
    total_distance.columns = ["Asset_ID", "Total_Distance"]

    # 🔹 평균 배송 시간 계산
    filtered_df["Delivery_Time"] = (filtered_df["Actual_Arrival_Time"] - filtered_df["Departure_Time"]).dt.total_seconds() / 3600  # 시간 단위 변환
    avg_delivery_time = filtered_df.groupby("Asset_ID")["Delivery_Time"].mean().reset_index()
    avg_delivery_time.columns = ["Asset_ID", "Avg_Delivery_Time"]
    
    # ✅ NULL 값 제거 후 병합
    truck_summary = shipment_count.merge(total_distance, on="Asset_ID").merge(avg_delivery_time, on="Asset_ID").dropna()

    # ✅ 🚛 트럭 ID 오름차순 정렬 (1 → 10 순서)
    truck_summary["Truck_Number"] = truck_summary["Asset_ID"].str.extract(r'(\d+)').astype(int)
    truck_summary = truck_summary.sort_values(by="Truck_Number", ascending=True).drop(columns=["Truck_Number"])

    
    # ✅ 📌 표 표시 (높이 조절 추가)
    st.dataframe(
        truck_summary.style.format({
            "Shipment_Count": "{:.0f}",
            "Total_Distance": "{:.2f} km",
            "Avg_Delivery_Time": "{:.1f} 시간"
        }),
        height=500, hide_index=True, use_container_width=True  # 📌 지도의 높이와 동일하게 조정
    )


# 📌 하단 3개 그래프 (넓게 활용)
with st.container():
    col1, col2, col3 = st.columns(3)

    
    with col1:
        st.markdown('<div class="title-box">🌡 온습도 모니터링</div>', unsafe_allow_html=True)

        # 📌 날짜 열 생성 (Timestamp에서 날짜만 추출)
        filtered_df["Date"] = filtered_df["Timestamp"].dt.date  

        # 📌 모든 날짜 범위 생성
        date_range = pd.date_range(start=filtered_df["Date"].min(), end=filtered_df["Date"].max())  
        all_dates_df = pd.DataFrame({"Date": date_range.date})  # DataFrame으로 변환

        # 📌 날짜별 평균 온도 및 습도 계산
        temp_humidity_avg = filtered_df.groupby("Date")[["Temperature", "Humidity"]].mean().reset_index()

        # 📌 전체 날짜 데이터와 병합 (빈 날짜도 포함)
        temp_humidity_avg = all_dates_df.merge(temp_humidity_avg, on="Date", how="left").fillna(0)

        # 📌 막대 그래프 생성 (x축: 날짜)
        fig2 = px.bar(temp_humidity_avg, x="Date", y=["Temperature", "Humidity"], barmode="group")

        # 📌 기준선 추가 (온도: 25도, 습도: 55%)
        fig2.add_hline(y=25, line_dash="dot", line_color="red", annotation_text="25°C", annotation_position="top left")
        fig2.add_hline(y=55, line_dash="dot", line_color="red", annotation_text="55%", annotation_position="top left")

        # 📌 x축 레이블 조정 (모든 날짜 포함)
        fig2.update_layout(
            xaxis_title="Date",
            yaxis_title="Value",
            plot_bgcolor="white",
            xaxis_tickangle=-90,
            xaxis=dict(
                tickmode='array',
                tickvals=temp_humidity_avg["Date"],
                ticktext=temp_humidity_avg["Date"].astype(str)
            ),
            showlegend=False  # 범례 제거
        )

        # 📌 그래프 표시
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.markdown('<div class="title-box">⏳ 물류 지연 원인별 분석</div>', unsafe_allow_html=True)

        # 데이터 복사 후 지연 여부 이진화
        df_copy = filtered_df.copy()
        df_copy["Logistics_Delay_Binary"] = (df_copy["Logistics_Delay"] > 0).astype(int)

        # 지연 원인 관련 칼럼
        cause_columns = ["Traffic_Status", "Weather_Condition", "Logistics_Delay_Reason"]

        # 모든 가능한 원인 항목을 담을 집합 (빈도수가 0인 항목도 포함)
        all_causes = set()
        for col in cause_columns:
            all_causes.update(df_copy[col].dropna().unique())

        # 각 원인이 지연 발생에 기여한 횟수 계산
        delay_counts = {}
        for col in cause_columns:
            cause_delay = df_copy[df_copy["Logistics_Delay_Binary"] == 1][col].value_counts()
            for cause, count in cause_delay.items():
                delay_counts[cause] = count

        # 모든 항목을 포함하도록 delay_counts에 없는 항목은 0으로 설정
        for cause in all_causes:
            delay_counts.setdefault(cause, 0)

        # 전체 지연 발생 횟수
        total_delays = sum(delay_counts.values()) if sum(delay_counts.values()) > 0 else 1  # 0으로 나누기 방지

        # 상위 n개 항목만 선택 (예: 상위 6개)
        top_n = 6  
        top_causes = sorted(delay_counts, key=delay_counts.get, reverse=True)[:top_n]

        # 기타 항목 처리
        filtered_delay_counts = {cause: delay_counts[cause] for cause in top_causes}
        filtered_delay_counts["기타"] = sum(delay_counts[cause] for cause in delay_counts if cause not in top_causes)

        # 데이터 변환
        labels = list(filtered_delay_counts.keys())  # X축 레이블
        sizes = [filtered_delay_counts[cause] / total_delays * 100 for cause in labels]  # 비율 계산

        # 진한 색상 팔레트 설정 (딥 블루, 오렌지, 레드, 퍼플, 그린, 옐로우)
        dark_colors = ['#1f77b4', '#4690d6', '#62a8e1', '#7fb5e5', '#99c2e6', '#b3d2e8', '#003366']

        # 색상 리스트 길이를 labels 길이에 맞게 확장
        dark_colors = dark_colors[:len(labels)]

        # Plotly로 세로 막대 그래프 생성
        fig = go.Figure(data=[go.Bar(x=labels, y=sizes, marker_color=dark_colors)])

        # 제목과 레이블 설정
        fig.update_layout(
            xaxis_title="Delay Causes",
            yaxis_title="Percentage (%)",
            plot_bgcolor='white',
            xaxis_tickangle=-90
        )

        # 그래프 표시
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        st.markdown('<div class="title-box">📊 시간대별 평균 지연 시간</div>', unsafe_allow_html=True)

        # ✅ Timestamp에서 "Hour" 컬럼 추출 후 평균 지연 시간 계산
        if "Hour" not in filtered_df.columns:
            filtered_df["Hour"] = filtered_df["Timestamp"].dt.hour

        delay_trend = filtered_df.groupby("Hour")["Delay_Time"].mean().reset_index()

        # ✅ 라인 그래프 생성
        fig5 = px.line(delay_trend, x="Hour", y="Delay_Time", markers=True)

        # x축과 y축 제목을 영어로 설정
        fig5.update_layout(
            xaxis_title="Hour",  # x축 제목: Hour
            yaxis_title="Average Delay Time (Minutes)",  # y축 제목: Average Delay Time (Minutes)
        )
        st.plotly_chart(fig5, use_container_width=True)
