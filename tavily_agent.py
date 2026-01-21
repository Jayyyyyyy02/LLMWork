"""
LangGraph 없이 직접 구현하는 반복 추론 Agent - TAVILY 검색
Agent의 핵심인 반복 추론(Reasoning Loop)을 직접 구현하여 동작 원리 이해
－－－－－－－－－－－－－
목표(goal)를 달성하기 위해 스스로 판단하고, 행동을 선택하고, 외부 도구를 사용할 수 있는 "자율적 실행자".
자가 판단 → 행동 선택 → 환경 반영 → 결과 분석
이 사이클을 ａｉ스스로 수행한다는 점이 핵심
Agent는 단순히 “답을 생성”하는 것이 아니라 목표(goal)를 달성하기 위해 단계별로 작업한다

Agent가 일할 때 핵심 기술이 바로 반복 추론이다.
이걸 다른 말로는 chain-of-thought, multi-step reasoning, self-reflection 등으로 부르기도 한다.
반복추론이란¿
큰 문제를 한 번에 해결하지 않고, 여러 단계로 쪼개서 순차적으로 해결하는 방식.
"""

import os 
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import SystemMessage, HumanMessage,ToolMessage

load_dotenv()

def main():
    '''반복 추론이 가능한 Agent 구현 및 실행'''
    openai_api_key = os.getenv('OPENAI_API_KEY')
    tavily_api_key = os.getenv('TAVILY_API_KEY')

    if not openai_api_key or not tavily_api_key:
        print('# api key가 없습니다 .env를 확인하세요')
        return
    
    # 1. 모델 생성
    llm = ChatOpenAI(model = 'gpt-4o-mini', temperature = 0, api_key = openai_api_key)
    # 2. 타빌리 검색 도구 생성
    search_tool = TavilySearchResults(
        max_results = 3,
        search_depth = "advanced",
        include_answer = True,
        include_raw_content = False,
        include_images = False,
        api_key = tavily_api_key
    )
    # 3. 도구 딕셔너리 (도구 빨리 찾기위해)
    tools = [search_tool]
    tool_dict = {"tavily_search_results_json": search_tool}

    # 4. llm모델에 도구 바인딩
    llm_with_tools = llm.bind_tools(tools)

    # 5. 복잡한 질문 실행(여러 단계 추론이 필요한 질문)
    questions = [
        "2025년 롤드컵 우승팀과 그들의 주료 선수를 알려줘",
        "비트코인 가격이 10만 달러 넘었는지 확인하고, 넘었다면 언제인지 알려줘"
    ]

    # 6. 반복추론을 실행하는 사용자 정의 함수 호출해서 결과 받기
    for q in questions:
        print('='*100)
        print(f'질문: {q}')
        print('='*100)
        run_agent(q, llm, llm_with_tools,tool_dict,max_iteration = 5)
        print('='*100)

def run_agent(question, llm, llm_with_tools, tool_dict, max_iteration = 5):
    """Agent 반복 추론 루프 (LangGraph 없이 직접 구현)
    
    이 함수는 Agent의 핵심 동작 원리를 보여줍니다:
    1. LLM이 현재 상황을 판단
    2. 도구가 필요하면 실행 → 다시 1번으로
    3. 충분한 정보를 얻으면 최종 답변
    
    Args:
        question (str): 사용자 질문
        llm (ChatOpenAI): 기본 LLM (최종 답변용)
        llm_with_tools (ChatOpenAI): 도구가 바인딩된 LLM
        tools_dict (dict): 사용 가능한 도구 딕셔너리
        max_iterations (int): 최대 반복 횟수 (무한 루프 방지)
    """
    # 메세지 히스토리 초기화
    messages = [
        SystemMessage(content = '''당신은 최신 정보를 검색하여 정확하게 답변하는 AI Agent입니다.

            주요 역할:
            - 질문에 답하기 위해 필요한 정보를 검색합니다.
            - 검색 결과가 불충분하면 추가 검색을 수행합니다.
            - 충분한 정보를 얻었다면 명확하고 자세한 최종 답변을 제공합니다.
            - 한국어로 답변합니다.

            도구 사용 전략:
            - 최신 정보가 필요하면 검색 도구를 사용하세요.
            - 검색 결과가 불완전하면 다른 키워드로 다시 검색하세요.
            - 여러 정보가 필요하면 순차적으로 검색하세요.
            - 충분한 정보가 모였다면 도구 호출 없이 최종 답변을 작성하세요.'''),
            HumanMessage(content = question)
    ]

    # 반복 추론 루프(Agent핵심)
    for i in range(max_iteration):
        print(f'[Iteration: {i + 1} / {max_iteration}]')
        print('-'*70)

        # LLM에 다음 행동 결정 요청
        response = llm_with_tools.invoke(messages)
        messages.append(response) # 대화 히스토리에 추가

        if response.tool_calls:
            # Agent가 더 정보가 필요해라고 판단한 경우
            print(f'Agent판단: 도구 사용 필요 {len(response.tool_calls)}개 도구 호')

            # 각 도구 실행
            for tool_call in response.tool_calls:
                tool_name = tool_call['name']
                tool_args = tool_call['args']

                print(f'도구명: {tool_name}, 도구 인자: {tool_args}')
                print(f'검색어: {tool_args.get('query', "None")}')

                # 도구 실행
                if tool_name in tool_dict:
                    try:
                        tool_output = tool_dict[tool_name].invoke(tool_args)
                        print(f'검색 완료: {len(tool_output)}개 결과 가져옴')

                        if tool_output:
                            first_result = tool_output[0].get('content','')[:100]
                            print(f'첫 결과: {first_result}')

                            # 도구 결과 히스토리에 추가
                            # LLM이 다음 반복에 이 정보를 참고한다
                            messages.append(ToolMessage(
                                content = str(tool_output),
                                tool_call_id = tool_call['id']
                            ))

                    except Exception as ex:
                        print(f'도구 실행 중 요류: {str(ex)}')
                        messages.append(ToolMessage(
                            content=f"오류 발생: {str(ex)}",
                            tool_call_id = tool_call['id']
                        ))
                else:
                    print(f'알수 없는 도구: {tool_name}')
            print(f"Agent가 다음 단계 분석 중")
        else:
            # 도구 호출이 없을 경우
            print('Agent판단: 충분한 정보 수집 완료 최종 답변 생성')
            print("="*70)
            print(response.content)
            print("="*70)
            return # Agent종료
    # 최대 반복 횟수에 도달한 경우
    print(f"최대 반복 횟수 ({max_iteration})에 도달했어요")
    print('마지막 상태로 답변을 생성합니다')

    # 최종 답변 요정
    final_response = llm.invoke(messages + [
        HumanMessage(content="지금까지 수집한 정보를 바탕으로 최종 답변을 작성해 주세요")
    ])
    print("="*70)
    print("최종 답변")
    print(final_response.content)
    print("="*70)


if __name__ == '__main__':
    main()

