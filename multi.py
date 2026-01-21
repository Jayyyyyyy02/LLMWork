from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
load_dotenv()

# 멀티턴 대화 예제
llm = ChatOpenAI(model = "gpt-4o-mini", temperature = 0.8)

# 시스템 메시지 : 페르소나 지정, task지정
persona = "너는 친절하고 상냥한 음식 상담사야 사용자의 질문과 취향을 이해하고 공감하며, 적절한 추천과 레시피를 제공한다"
system_msg = SystemMessage(content = persona)

# 대화 히스토리를 초기화
message = [system_msg,]

# 멀티턴 질문 함수 정의
def ask_user(question):
    """
    사용자가 질문을 받아서 LLM모델에 전달하고 , 응답을 반환하는 함수
    내부적으로 message 리스트에 대화 히스토리르 누적하여 멀티턴을 유지한다
    """
    # 질문 추가
    message.append(HumanMessage(content = question))

    # LLM모델 호출
    response = llm.invoke(message)

    # 응답을 AIMessage객체에 담아서 messages에 추가
    message.append(AIMessage(content = response.content))
    return response.content

# 멀티턴 대화 테스트
import time
q1 = "내가 좋아하는 음식은 달콤한 음식이야"
reply1 = ask_user(q1)
print("User 질문 1 : ", q1)

time.sleep(1)
print("모델 응답 1 : ", reply1)
print("*"*100)

q2 = "그럼 달콤한 음식 중에 추천 메뉴를 알려줘"
reply2 = ask_user(q2)
print("User 질문 3 : ", q2)

time.sleep(1)
print("모델 응답 2 : ", reply2)
print("*"*100)

q3 = "추천한 메뉴 중 하나 골라 레시피를 자세히 설명해줘"
reply3 = ask_user(q3)
print("User 질문 3 : ", q3)

time.sleep(1)
print("모델 응답 3 : ", reply3)
print("*"*100)

# 대화 히스토리 확인하기
print("전체 대화 히스토리 : ")
for idx, msg in enumerate(message):
    print(f'{idx + 1} - {msg.type} : {msg.content[:100]}...')