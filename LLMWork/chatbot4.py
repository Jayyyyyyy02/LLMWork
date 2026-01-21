import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

# 1. api key 로드
load_dotenv()

# 2. AI모델 생성
llm = ChatOpenAI(model = "gpt-4o-mini", temperature = 0.7)

# 3. 스트림릿 기본 설정
st.set_page_config(page_title = "AI챗봇1-Basic", layout = "centered")
st.header("😻기본 챗봇 (langchain+streamlit)")
st.caption("PromptTemplate + 싱글턴 대화")

# 4. 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state['messages'] = [] #Human/AI 메시지 저장할 예정

# 5. PromptTemplate으로 템플릿 생성
prompt = PromptTemplate(input_variables=["question"],
            template = "너는 친절한 상담 챗봇이야 사용자 질문에 명확하고 친절하게 답변해줘\n질문: {question}")

# 6. 세션 상태에 저장된 기존 메시지가 있으면 출력
for msg in st.session_state['messages']:
    if isinstance(msg,HumanMessage): #사용자 메시지 출력
        with st.chat_message("user"):
            st.write(msg.content)
    else: #모델 메시지 출력
        with st.chat_message("assistant"):
            st.write(msg.content)
# 7. 사용자 입력=>세션 상태에 내 입력 메시지 저장
user_input = st.chat_input("메시지를 입력하세요")
if user_input:
    # 사용자 메시지 저장
    human_msg = HumanMessage(content = user_input)
    st.session_state['messages'].append(human_msg)

    with st.chat_message("user"):
        st.write(user_input)
    # PromptTemplate 질문 적용
    final_prompt = prompt.format(question=user_input)

    # 8. 모델 호출해서 응답 받기
    with st.chat_message("assistant"):
        with st.spinner("답변 생성 중..."):
            response = llm.invoke([HumanMessage(content = final_prompt)])
            st.write(response.content)

# 9. 세션 상태에 응답 내용 저장
    st.session_state['messages'].append(response)