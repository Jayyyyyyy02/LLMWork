# Few-shot learning 예제
# system : "너는 요리 전문가이자, 백종원 대표의 말투를 흉내내는 AI야. 
# 충청도 사투리와 친근한 말투로 대답해"
# 예시1
# ("user","된장찌개 끓이는 법 알려줘")
# ("assistant","된장찌개는 말이여, 우선 재료를 준비해야혀. 된장, 두부, 애호박, 감자, 양파, 대파, 마늘, 고추 등을 준비하고...")

# 예시2 ...

# 새로운 요청(질문)

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

llm = ChatOpenAI(model = "gpt-4o-mini", temperature = 0.8)

prompt = ChatPromptTemplate.from_messages([
    ("system","너는 요리 전문가이자, 백종원 대표의 말투를 흉내내는 AI야. 충청도 사투리와 친근한 말투로 대답해"),

    # [예시 1]
    ("user","된장찌개 끓이는 법 알려줘"),
    ("assistant","된장찌개는 말이여, 우선 재료를 준비해야혀. 된장, 두부, 애호박, 감자, 양파, 대파, 마늘, 고추 등을 준비하고..."),
    # [예시 1]
    ("user","김치찌개 끓이는 법도 알려줘"),
    ("assistant","김치찌개는 말이여, 우선 신김치가 있어야혀. 돼지고기, 김치, 두부, 양파, 대파, 마늘, 고춧가루 등을 준비하고 김치는 묵은지를 써야혀. 그래야 깊은 맛이 난다잉"),
    # [새로운 질문]
    ("user","{question}"),
])

# LCEL 표현 (LangChain Expression Language) 파이프라인 연결
output_str = StrOutputParser()
chain = prompt | llm | output_str

# 질문 입력받기
user_question = input("요리 관련 질문을 입력하세요: ")

if not user_question.strip():
    print("질문이 입력되지 않았습니다. 프로그램을 종료합니다.")
    exit()

# 체인 실행
result = chain.invoke({"question" : user_question})

print("*"*100)
print(result)