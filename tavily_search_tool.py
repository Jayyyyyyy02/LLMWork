import os
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import SystemMessage,HumanMessage,ToolMessage

load_dotenv()

tavily_api_key = os.getenv('TAVILY_API_KEY')
if not tavily_api_key:
    print('tavily api key 가 없어요 .env를 확인하세요')

# 1.모델 초기화
llm=ChatOpenAI(model = 'gpt-4o-mini', temperature = 0)

# 2. 타빌리 검색 도구 설정
# TavilySearchResults 생성==>tools 리스트에 추가
# 
tavily_tool = TavilySearchResults(
    max_results = 3,
    api_key = tavily_api_key,
    search_depth = "advanced", #검색 깊이(basic:기본, advanced:상세 검색)
    include_answer = True, # 타빌리가 생성한 요약 답변 포함 여부
    include_images = False # 이미지 url포함 여부
)
print(tavily_tool.name) # 툴 이름 : tavily_search_results_json
tools = [tavily_tool]
tools_dict = {tavily_tool.name:tavily_tool}

# 3. llm에 도구 바인딩
llm_with_bind = llm.bind_tools(tools)
print("LLM에 tool binding완료!!")
# 4. 질문 여러개 리스트에 담기
questions = [
    "2025년 롤드컵 우승자를 알려줘",
    "2026년 1월 현재 비트코인 가격은?",
    "최신에 발표된 휴머노이드 로봇에 대해 알려줘"
]
# messages=[SystemMessage]
template = """
당신은 사용자 질문에 답변하는 assistant입니댜
최신 정보를 검색할 때는 타빌리 검색 도구를 사용하세요
"""

#메시지 컨텍스트 준비

# 각 질문에 대해 순차적으로 처리하는 반복문
# [1]llm_with_tools를 이용해 질문하기
# [2] 호출해야할 툴을 tool_calls에서 찾아 툴 호출하기
# [3] 그 결과 ToolMessage객체로 만들기=> messages에 append
for q in questions:
    print("===================="*2)
    print(f"질 문: {q}")
    print("===================="*2)
    messages = [SystemMessage(content = template)]
    messages.append(HumanMessage(content = q))

    response=llm_with_bind.invoke(messages)
    print('response: ', response)
    #########################
    messages.append(response)
    ##########################
    #툴 호출 정보 확인
    tool_calls = response.tool_calls
    if not tool_calls:
        #도구 호출이 필요 없다면=> 바로 답변 출력
        print("AI 최종 답변: ",response.content)
        continue

    #도구 호출이 필요한 경우
    for tool in tool_calls:
        tool_name = tool['name']
        tool_args = tool['args']
        print(f"tool_args: {tool_args}")
        search_tool = tools_dict.get(tool_name)
        try:
            tool_result = search_tool.invoke(tool_args) 
            # tool_result => list임
            print(f"{len(tool_result)}개 검색되었어요*****")
            #타빌리 검색 결과가 list임. 이를 문자열로 변환해서 ToolMessage에
            #넣어야 한다
            tool_msg = ToolMessage(
                content = str(tool_result),#########               
                tool_call_id = tool['id'] 
            )
            #messages.append(response) # llm의 tool_calls 메시지 저장
            messages.append(tool_msg) #도구 결과 저장

        except Exception as ex:
            print(f"도구 호출 실행 중 오류: ",ex)
            messages.append(ToolMessage(content = "오류 발생: " + str(ex),
                                        tool_call_id = tool['id']))

        #도구 결과 기반으로 최종 답변 생성
        final_response = llm_with_bind.invoke(messages)
        print("="*100)
        print('최종 답변:')
        print("="*100)
        print(final_response)

        #마지막 답변도 저장
        messages.append(final_response)

#5. 도구 결과 (messages)를 바탕으로 llm을 호출하여 최종 답변 생성후 출력하기