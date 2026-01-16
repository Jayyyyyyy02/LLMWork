import streamlit as st

# 1. 페이지 설정 (브라우저 탭이름 아이콘 설정)
st.set_page_config(page_title = "나의 스트림릿", page_icon = "💖")

# 2. 사이드바(sidebar) : 설정이나 메뉴 배치
with st.sidebar:
    st.header("🛠️ 설정")
    user_color = st.color_picker("테마 색상을 선택하세요","#55abcd")
    user_speed = st.slider("학습 속도 조절",0,100,50)

    st.info(f"선택한 색상: {user_color} 학습 속도:{user_speed}")

# 3. 메인 타이틀
st.title("🚩멋진 대시보드") #h1
st.subheader("정말 멋져요") #h2

# 4. 탭 : 내용 분할
tab1, tab2, tab3 = st.tabs(["소개","데이터 분석","🤖 AI 챗봇"])

with tab1:
    st.subheader("안녕하세요?")
    st.write("여기는 첫 번째 탭입니다. 사이드바의 설정을 바꿔보세요")
    # 5. 컬럼(columns)이용해서 화면 가로 분할
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label = "배터리 잔량", value = "80%", delta = "-10%")
    with col2:
        st.metric(label = "현재 속도", value = f"{user_speed} km/h",  delta = "1.2 km/h")    
with tab2:
    st.subheader("데이터 시각화")
    st.info("여기에 차트나 표를 넣을 수 있어요")
    # 확장기(expander) : 상세내용 숨기거나 보여주기
    with st.expander("도움말 보기"):
        st.write("탭과 컬럼을 조합하면 복잡한 화면이 정리돼요")

# tab3를 추가해서
# AI챗봇
# 채팅 ui 구현해보기(제목/텍스트박스/메시지출력)

with tab3:
    st.subheader("::AI챗봇::")
    st.write("반가워요~ 저는 여러분의 친구 AI도우미입니다")

    msg = st.text_input("내용을 입력하세요: ")
    # msg내용을 LLM 모델에 전달하여 받은 응답을 출력
    if msg:
        st.write(msg)
        