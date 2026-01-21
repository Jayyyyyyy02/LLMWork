from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
load_dotenv()

# 1. 모델 선언
llm = ChatOpenAI(model = "gpt-4o-mini", temperature = 0.7)

# 2. 메모리 객체 선언(윈도우 크기 지정)
memory = ConversationBufferWindowMemory(k = 2, return_messages = True)
# k = 2 : 최근 2턴의 대화만 메모리에 저장

# 3. 프롬프트 템플릿 작성
prompt = ChatPromptTemplate.from_messages([
    ("system", "너는 친절한 독서 전문가야. 사용자 취향에 맞춰서 책을 추천하고, 읽기 계획도 구체적으로 제안해줘"),
    MessagesPlaceholder(variable_name = "history"), # 이전 대화 삽입
    ("human", "{input}"), # 사용자 질문
])

# 4. LCEL표현(프롬프트 -> LLM모델 -> 메모리) 파이프라인 연결
chain = prompt | llm

# 연속 대화 시뮬레이션

inputs = [
    "이번 달에 읽을 책 2권 추천해줘.",
    "그럼 한 달 동안 2권을 모두 읽을 수 있는 주간 독서 계획도 만들어줄래?",
    "그 계획에 따라 매일 읽을 페이지 수도 알려줄 수 있어?",
    "내가 아까 추천해달라고 한 책들은 무엇이었지?"
]

for user_input in inputs:
    # 메모리 불러오기(최근 대화 2개만 포함)
    history = memory.load_memory_variables({})["history"]

    # 체인 실행
    result = chain.invoke({"history" : history, "input" : user_input})

    # 결과 출력 
    print(f"\n사용자 : {user_input}") 
    print(f"AI응답 : {result.content}")
    print("*"*50)

    # 메모리에 저장
    memory.save_context({"input" : user_input},{"output" : result.content})

    # 이 방식은 단기 기억처럼 동작한다
    # 오래된 대화는 자동으로 메모리에서 지워져 토큰 낭비를 줄이고 호율성을 높인다