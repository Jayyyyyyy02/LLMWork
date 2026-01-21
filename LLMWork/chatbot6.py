# 사용자별 멀티 세션 관리 : RunnableWithMessageHistory + ChatPromptTemplate
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from langchain_core.runnables.history import RunnableWithMessageHistory
# 히스토리 관리

from langchain_community.chat_message_histories import ChatMessageHistory
# 메시지 저장소

load_dotenv()

# 1. 스트림릿 기본 설정
st.header("세션별 멀티턴 챗봇")
st.subheader("RunnableWithMessageHistory + ChatPromptTemplate")

if "store" not in st.session_state:
    st.session_state.store = {}
    # store : 세션 ID별로 대화 히스토리를 저장하는 딕셔너리

# 2. 세션별 히스토리를 변환하는 
def get_session_history(session_id : str):
    """
    세션별 ChatMessageHistory 반환
    - session_id : 사용자/대화 세션을 구분하는 고유 ID
    - 해당 세션의 히스토리가 없으면 새로 생성
    """
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id] # ChatMessageHistory객체 반환

session_id_input = st.sidebar.text_input("사용자 세션 ID 입력", value = "user_1")
SESSION_ID = "hong"
# 사용자가 로그인한 아이디 정보를 할당할 예정

# 3. 모델 생성
llm = ChatOpenAI(model = "gpt-4o-mini", temperature = 0.7)

# 4. ChatPromptTemplate 구성
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "너는 친절한 여행 도우미야. 사용자 질문에 구체적이고 현실적으로 답변해줘."),
    MessagesPlaceholder(variable_name = "history"),
    #이전 대화가 여기 삽입됨
    ("human","{input}") # 사용자 질문이 삽입됨
])

# 5. 체인 구성 (LCEL)
chain = chat_prompt | llm

# 6. RunnableWithMessageHistory : 히스토리 자동 관리 래퍼
# - chain 실행할 체인
# - get_session_history : 세션별 히스토리를 가져오는 함수
# - input_messages_key : 사용자 입력이 들어갈 키 이름(프롬프트의 {input}과 매칭)
# - history_messages_key : 히스토리가 들어갈 키 이름(프롬프트의 MessagesPlaceholder와 매칭)

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key = "input", # {input: "질문"}
    history_messages_key = "history" # 이전 대화르 history에 자동 삽입
)

# 7. ui 구성 대화 메시지 화면에 출력
history = get_session_history(SESSION_ID) # 현재 세션의 대화 기록 가져오기

for msg in history.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.write(msg.content) # 메시지 내용 출력

# 8. 사용자 입력 받기 
user_input = st.chat_input("메시지를 입력하세요")

if user_input:
    with st.chat_message("user"):
        st.write(user_input)
    # 모델 호출
    # config => 세션 아이디 전달하여 어떤 히스토리 사용할지 지정
    config = {"configurable" : {"session_id": SESSION_ID}}

    # AI 응답
    with st.chat_message("assistant"):
        with st.spinner("생각 중..."):
            response = chain_with_history.invoke(
                {"input" : user_input}, # 사용자 질문
                config = config # 세션 설정
            )
            st.write(response.content) # 응답 표시
