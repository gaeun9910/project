import streamlit as st

# 1. 여러 페이지 정보를 등록합니다.
pages = [
    st.Page("project_1.py", title="이차전지 물류 대시보드", icon="🔋", default=True),
    st.Page("project_2.py", title="이차전지 물류 지연 시간 예측", icon="📦"),
]

# 2. 사용자가 선택한 페이지를 받아옵니다.
selected_page = st.navigation(pages)

# 3. 선택된 페이지를 실행(run)합니다.
selected_page.run()
