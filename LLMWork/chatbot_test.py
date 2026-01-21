from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.memory import ConversationSummaryMemory


load_dotenv()
llm = ChatOpenAI(model = "gpt-4o-mini", temperature = 0.7)

memory = ConversationSummaryMemory(llm=llm, return_messages=True)
parser =  StrOutputParser()

# 프롬프트 템플릿 작성
prompt = ChatPromptTemplate.from_messages([
    ("system",
    "당신은 게임 메카닉 설명 전문 챗봇 개발자입니다.게임을 처음 시작하는 초보자들이 게임 용어를 물어보면, 챗봇은 다음 3가지를 포함한 답변을 해야 합니다"),
    MessagesPlaceholder(variable_name = "history"), # 이전 대화 삽입
    ("human","{input}"), # 사용자 질문
])

# 체인 구성
chain = prompt | llm |parser

# 연속 대화 시뮬레이션
inputs = [
        "넉백이 뭐야?",
        "메타란?",
        "프레임 드롭이 뭐야?",
        "핑이란?",
        "크리티컬이 뭐야?",
        "쿨타임이란?",
        "너프가 뭐야?",
]
for user_input in inputs:
    # 메모리 불러오기 (요약된 대화 포함)
    history = memory.load_memory_variables({})["history"]
    # 체인 실행
    result = chain.invoke({"history": history, "input": user_input})

    # 결과 출력
    print(f"\n사용자 : {user_input}")
    print(f"AI응답: {result}")
    print("*"*50)
    # 메모리에 저장
    memory.save_context({"input": user_input}, {"output": result})
