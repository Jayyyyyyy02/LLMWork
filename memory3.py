from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_classic.memory import ConversationSummaryMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

# 1. 모델 선언
llm = ChatOpenAI(model = "gpt-4o-mini", temperature = 0.7)

# 2. 메모리 객체 선언 (요약 메모리)
memory = ConversationSummaryMemory(llm=llm, return_messages=True)

# 3. 프롬프트 템플릿 작성
prompt = ChatPromptTemplate.from_messages([
    ("system",
    "너는 친절한 독서 전문가야. 사용자 취향에 맞춰서 책을 추천하고, 읽기 계획도 구체적으로 제안해줘. 메모리 요약본은 한국어로 보관해줘"),
    MessagesPlaceholder(variable_name = "history"), # 이전 대화 삽입
    ("human","{input}"), # 사용자 질문
])
# 4. 체인 구성
chain = prompt | llm

# 5. 연속 대화 시뮬레이션
inputs = [
    "이번 달에 읽을 책 2권 추천해줘. 예) 나는 파친코 같은 책 좋아해 소설류로 추천해줘",
    "추천한 책 한 달 동안 2권을 모두 읽을 수 있는 주간 독서 계획도 만들어줄래?",
    "대한민국 수도는 어디야?",
    "서울의 인구는 몇 명이야?",    
    "내가 아까 추천해달라고 한 책들은 무엇이었지?"
]
for user_input in inputs:
    # 메모리 불러오기 (요약된 대화 포함)
    history = memory.load_memory_variables({})["history"]
    # 체인 실행
    result = chain.invoke({"history": history, "input": user_input})

    # 결과 출력
    print(f"\n사용자 : {user_input}")
    print(f"AI응답: {result.content}")
    print("*"*50)
    # 메모리에 저장
    memory.save_context({"input": user_input}, {"output": result.content})

# 6. 현재까지의 요약본 확인
print("\n=== 대화 요약본 ======================================")
print(memory.buffer) # 요약할 때 영어로 요약하도록 설계되어 있음