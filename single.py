from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
load_dotenv()

# system : SystemMessage객체
# user : HumanMessage객체
# assistant : AIMessage객체

llm = ChatOpenAI(model = 'gpt-4o-mini', temperature = 0.7)

# 단일턴 대화 예제
user_input1 = "내 이름은 swan이야"
user_input2 = "내 이름이 뭐라고 했지?"

import time

response1 = llm.invoke([
    HumanMessage(content = user_input1)
])
print("모델 응답1 : ", response1.content)

time.sleep(1) # 1초 대기

response2 = llm.invoke([
    HumanMessage(content = user_input2)
])
print("모델 응답2 : ", response2.content)