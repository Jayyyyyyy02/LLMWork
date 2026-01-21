# 메시지 입력 후 자동으로 input 비우기
import streamlit as st
st.title("::SMU 채팅::")
st.write("Welcome to SMU World~")

with st.form("msg_form", clear_on_submit = True):
    msg = st.text_input("대화 내용을 입력하세요")

    submitted = st.form_submit_button("보내기")
    # 보내기 버튼을 누를 때만 form이 전송되고 rerun이 발생함
    # form 내부에 모든 입력창들이 초기화 된다

# submit한 후 화면에 메시지 출력하기
if submitted:
    st.write(f"Ehco>>{msg}")