from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# 1. 모델 선언
llm = ChatOpenAI(model = 'gpt-4o-mini', temperature = 0.7)

# 2. 프롬프트 템플릿 작성
# - system : 역할(persona)지정
# - user : 사용자 질문
# - assistant : ai 응답

prompt = ChatPromptTemplate.from_messages([
    ("system", "너는 친절한 독서 전문가야. 사용자 취향에 맞춰서 책을 추천하고, 읽기 계획도 구체적으로 제한해줘"),
    ("user", "이번 달에 읽을 만한 책 3권 추천해줘"),
    ("assistant", "좋아! 어떤 장르를 선호하니?"),
    ("user", "나는 SF와 판타지를 좋아해")
])
output_str = StrOutputParser()

# 3. 출력 파서 선언
output_str = StrOutputParser()

# 4. LCEL표현 (프롬프트 -> LLM모델 -> 출력) 파이프라인 연결
chain = prompt | llm | output_str

# 5. 실행
result = chain.invoke({})

# 6. 결과 출력
print(result)