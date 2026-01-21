from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
load_dotenv()

# 1. 모델 선언
llm = ChatOpenAI(model = "gpt-4o-mini", temperature = 0.7)

# 2. 메모리 객체 선언
memory = ConversationBufferMemory(return_messages = True)

# 3. 프롬프트 템플릿 작성
prompt = ChatPromptTemplate.from_messages([
    ("system", "너는 친절한 독서 전문가야. 사용자 취향에 맞춰서 책을 추천하고, 읽기 계획도 구체적으로 제안해줘"),
    MessagesPlaceholder(variable_name = "history"), # 이전 대화 삽입
    ("human", "{input}"), # 사용자 질문
])

#MessagesPlaceholder는 프롬프트 템플릿 안에 동적으로 메시지 리스트를 삽입하는 자리 표시자
#variable_name="history"로 지정된 변수에 저장된 이전 대화 메시지들을 그대로 삽입
#메모리에서 불러온 대화 내역이 이 자리에 들어감

# 4. LCEL표현(프롬프트 -> LLM모델 -> 메모리) 파이프라인 연결
chain = prompt | llm

# 5. 여러 질문을 리스트에 저장
inputs = [
    "이번 주 읽을 만한 책 2권 추천해줘",
    "그럼 이번 주에 2권 모두 읽을 수 있게 주간 독서 계획표 만들어줘"
]

# 6. 반복문을 통해 체인 실행
for user_input in inputs:
    # 메모리에서 대화 히스토리 가져오기
    history = memory.load_memory_variables({})["history"]

    # 체인 실행 => 이때 history와 input값을 전달 
    result = chain.invoke({"history" : history, "input" : user_input})

    # 결과 출력 
    print(f"\n사용자 : {user_input}") 
    print(f"AI응답 : {result.content}")

    # 메모리에 저장
    memory.save_context({"input" : user_input},{"output" : result.content})