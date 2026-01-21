from dotenv import load_dotenv
from datetime import datetime
import pytz

from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
import os

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
if api_key:
    print(api_key[:5])

else:
    print('***api key가 없습니다***')

# 도구(Tool) 정의 도구를 정의할 떄는 description 정보가 중요 => llm 파악할 수 있는 정보
@tool # @tool decorator 를 붙인다 => 파이썬 함수를 랭체인이 인식할 수 있는 '도구'로 변환함 
def get_current_time(timezone: str, location: str) -> str:
    '''
    현재 시간을 알려주는 tool입니다 
    Args:
        timezone(str) : 타임존(예: 'Asia/Seoul')
        location(str) : 지역명(예: 서울)
    Returns:
    str: 'Asia/Seoul(Seoul) 현재 시각 YYYY-MM-DD HH:MM:SS'형식 문자열
    '''
    try:
        tz = pytz.timezone(timezone)
        now = datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S')
        loc_local_time = f'{timezone} ({location} 현재 시각{now})'
        print(f'***[Tool 실행 결과] : {loc_local_time}***')
        return loc_local_time
    except pytz.exceptions.UnknownTimeZoneError:
        return f'오류: 알 수 없는 타임존{timezone} 입니다'

# 메인 실행 로직--------------------
def run_tool():
    """
    랭체인 툴과 openAI 모델을 연동하여 실행하는 함수 
    """
    # 1. 모델 초기화
    llm = ChatOpenAI(model = "gpt-4o-mini", temperature = 0)
    # 툴 호출시에는 정확도 중요하므로 temperature를 0으로 설정한다

    # 2. 도구 바인딩 : llm모델에게 사용할 수 있는 도구(tools) 목록을 알려준다
    tools = [get_current_time] # 여러 툴들을 관리할 리스트 생성
    tool_dict = {"get_current_time": get_current_time}
    # 실제 함수 실행을 위해 딕셔너리에 키값과 툴을 매핑하여 저장한다
    # 바인딩
    llm_with_tools = llm.bind_tools(tools)

    # 메시지 히스토리 관리
    messages = [
        SystemMessage("당신은 사용자 질문에 답변하기 우해 제공된 tools를 사용할 수 있습니다"),
        HumanMessage("서울은 지금 몇 시야?")
    ]
    print("\n" + "*"*50)
    print("1차 호출: 모델이 툴 사용을 결정합니다")
    print(f'User Question: {messages[-1].content}')
    print("*"*50)

    # 바인딩된 모델로 호출
    response = llm_with_tools.invoke(messages)
    print(f"Model Response1 (Tool Call Intention) : {response}")
    messages.append(response)
    # AI응답을 메시지에 추가 => '도구 호출 의도'를 메시지 히스토리에 추가
    # llm이 사용자 질문에 대해 tool을 호출할 필요가 있다고 판단하고 tool_calls 정보를 넘긴다. 
    # 이 정보를 분석해서 파이썬이 해당 함수를 호출하도록 해야 한다
    '''
    tool_calls=[
    {'name': 'get_current_time',
    'args': {'timezone': 'Asia/Seoul', 'location': '서울'},
    'id': 'call_jJgt4n4JfqY7MpTbx5Hl4zrq', 
    'type': 'tool_call'
    }] 
    '''

    # [2] 실제 툴 실행 로직
    if response.tool_calls:
        print("*"*50)
        print('2차 호출: 툴을 실해아고 모델에 그 결과를 다시 전달')
        print("*"*50)
        for tool_call in response.tool_calls:
            tool_name = tool_call['name'] # get_current_time
            tool_args = tool_call['args'] # {'timezone': 'Asia/Seoul', 'location': '서울'}

            if tool_name in tool_dict:
                select_tool = tool_dict[tool_name] # => 함수를 반환
                print(f"[Tool Call]: {tool_name} 호출 예정. 매개변수: {tool_args}")
                # 실제 파이썬 함수 실행
                tool_output = select_tool.invoke(tool_args)

                # [3단계] 툴 호출햐여 받은 결과를 ToolMessage객체에 담는다
                # => 중요. 이때 어떤 도구 호출에 대한 결과인지 명시해야
                # 모델이 매칭할 수 있다.(id값을 전달해야 함)
                tool_msg = ToolMessage(
                    content = tool_output,
                    tool_call_id = tool_call['id']
                )
                messages.append(tool_msg)
            else:
                print(f"경고: 정의되지 않은 툴 {tool_name}이 호출되었어요")
        # [4단계] 모델에게 messages를 전달하여 다시 호출
        # mesages(히스토리)에는 [사용자 질문 -> ai도구 호출 의도]-> 실제 도구 실행 결과]
        # 가 포함되어 있음
        print("*"*50)
        print('3차 호출: 툴 실행 결과를 바탕으로 최종 답변을 생성합니다')
        print("*"*50)
        final_response = llm_with_tools.invoke(messages)
        print(f"Final Model Response: {final_response.content}")
    else:
        # 도구가 필요 없는 질문일 경우 바로 답변 출력
        print("[결과]: 모델이 툴 호출을 결정하지 않아 바로 답변을 반환했습니다")
        print(f"AI 모델 답변: {response.content}")               
if __name__ == '__main__':
    run_tool()