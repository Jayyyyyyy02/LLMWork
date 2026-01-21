import streamlit as st

st.title("::나의 첫 채팅::")
st.write("안녕하세요? 저는 여러분의 챗봇 친구입니다")

# 텍스트 입력 위젯
name = st.text_input("이름을 입력하세요: ")

if name:
    st.write(f"안녕하세요, {name}님 만나서 반가워요~")