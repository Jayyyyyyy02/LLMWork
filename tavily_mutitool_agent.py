"""
다중 도구 Agent - TAVILY 검색 + Python 계산기
여러 도구 중 상황에 맞는 도구를 선택하고 조합하는 Agent 예제
"""

# 필요한 라이브러리 임포트
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

# .env 파일에서 환경 변수 로드
load_dotenv()


# 커스텀 도구 정의: Python 계산기
@tool
def python_calculator(expression: str) -> str:
    """Python 표현식을 안전하게 계산합니다.
    
    사용 예시:
    - "100 * 2" → 200
    - "(500 - 100) / 100 * 100" → 400.0
    - "pow(2, 10)" → 1024
    
    Args:
        expression: 계산할 Python 표현식 (문자열)
    
    Returns:
        계산 결과 (문자열)
    """
    try:
        # 안전한 수학 연산만 허용 (eval 대신 제한된 환경 사용)
        allowed_names = {
            'abs': abs, 'round': round, 'min': min, 'max': max,
            'sum': sum, 'pow': pow, 'len': len,
        }
        
        # eval을 안전하게 사용 (제한된 함수만 허용)
        result = eval(expression, {"__builtins__": {}}, allowed_names)
    # {"__builtins__": {}} : 아무런 기본 함수를 사용할 수 없도록 강제로 차단
        return str(result)
    except Exception as e:
        return f"계산 오류: {str(e)}"


def main():
    """다중 도구를 사용하는 Agent 구현 및 실행"""
    
    # API 키 로드
    openai_api_key = os.getenv("OPENAI_API_KEY")
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    
    if not openai_api_key or not tavily_api_key:
        print(" # API 키를 찾을 수 없습니다. .env 파일을 확인해주세요.")
        return
    
    print(" # API 키 로드 완료\n")
    
    # 1. LLM 모델 초기화
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        api_key=openai_api_key
    )
    print(" # OpenAI LLM 초기화 완료")
    
    # 2. 도구들 설정
    # 2-1. TAVILY 검색 도구
    search_tool = TavilySearchResults(
        max_results=3,
        search_depth="advanced",
        include_answer=True,
        include_raw_content=False,
        include_images=False,
        api_key=tavily_api_key
    )
    
    # 2-2. Python 계산기 도구
    calculator = python_calculator
    
    # 모든 도구를 리스트로 관리
    tools_list = [search_tool, calculator]
    
    # 도구 딕셔너리 (도구 이름으로 빠르게 찾기 위함)
    tools = {
        "tavily_search_results_json": search_tool,
        "python_calculator": calculator,
    }
    
    print(" # 도구 설정 완료:")
    print("   - TAVILY 검색 도구")
    print("   - Python 계산기 도구")
    
    # 3. LLM에 모든 도구 바인딩
    llm_with_tools = llm.bind_tools(tools_list)
    print(" # Agent 초기화 완료 (2개 도구 바인딩)\n")
    
    # 4. 다중 도구 조합이 필요한 복잡한 질문 실행
    questions = [
        "2025년 비트코인 최고가와 2020년 비트코인 평균 가격을 비교해서, 몇 퍼센트 상승했는지 계산해줘",
        "2025년 롤드컵 우승팀의 우승 횟수와 준우승팀의 우승 횟수를 더하면?",
        "단순 계산: 1234 곱하기 5678은?",
    ]
    
    for question in questions:
        print(f"\n{'='*70}")
        print(f"질문: {question}")
        print(f"{'='*70}\n")
        
        run_multi_tool_agent(question, llm, llm_with_tools, tools, max_iterations=7)
        print("\n" + "="*70 + "\n")


def run_multi_tool_agent(question, llm, llm_with_tools, tools, max_iterations=7):
    """다중 도구를 사용하는 Agent 반복 추론 루프
    
    이 함수는 여러 도구 중 상황에 맞는 도구를 선택하고 조합하는 방법을 보여줍니다.
    
    Args:
        question (str): 사용자 질문
        llm (ChatOpenAI): 기본 LLM
        llm_with_tools (ChatOpenAI): 도구가 바인딩된 LLM
        tools (dict): 사용 가능한 도구 딕셔너리
        max_iterations (int): 최대 반복 횟수
    """
    
    # 메시지 히스토리 초기화
    messages = [
        SystemMessage(content="""당신은 검색과 계산을 수행할 수 있는 AI Agent입니다.

        사용 가능한 도구:
        1. tavily_search_results_json: 최신 정보를 웹에서 검색합니다.
        2. python_calculator: 수학 계산을 수행합니다.

        주요 역할:
        - 질문을 분석하여 필요한 도구를 선택합니다.
        - 검색이 필요하면 검색 도구를, 계산이 필요하면 계산기를 사용합니다.
        - 여러 도구를 순차적으로 조합하여 복잡한 문제를 해결합니다.
        - 한국어로 답변합니다.

        도구 사용 전략:
        - 최신 정보나 실시간 데이터가 필요하면 검색 도구를 사용하세요.
        - 숫자 계산이 필요하면 계산기 도구를 사용하세요.
        - 검색 결과의 숫자를 계산해야 한다면 검색 후 계산기를 사용하세요.
        - 한 번에 하나씩 단계적으로 처리하세요.
        - 충분한 정보와 계산이 완료되면 최종 답변을 작성하세요."""),
        HumanMessage(content=question)
    ]
    
    # 반복 추론 루프
    for iteration in range(max_iterations):
        print(f"\n[Iteration {iteration + 1}/{max_iterations}]")
        print("-" * 70)
        
        # Step 1: LLM에게 다음 행동 결정 요청
        ai_msg = llm_with_tools.invoke(messages)
        messages.append(ai_msg)
        
        # Step 2: 도구 호출이 있는지 확인
        if hasattr(ai_msg, 'tool_calls') and ai_msg.tool_calls:
            print(f"🔧 Agent 판단: 도구 사용 필요 ({len(ai_msg.tool_calls)}개 도구 호출)")
            
            # Step 3: 각 도구 호출 실행
            for tool_call in ai_msg.tool_calls:
                tool_name = tool_call['name']
                tool_args = tool_call['args']
                
                print(f"\n   # 선택된 도구: {tool_name}")
                
                # 도구별로 다른 정보 출력
                if tool_name == "tavily_search_results_json":
                    print(f"  📝 검색어: {tool_args.get('query', 'N/A')}")
                elif tool_name == "python_calculator":
                    print(f"  🧮 계산식: {tool_args.get('expression', 'N/A')}")
                else:
                    print(f"  📝 인자: {tool_args}")
                
                # 도구 실행
                if tool_name in tools:
                    try:
                        tool_output = tools[tool_name].invoke(tool_args)
                        
                        # 결과 요약 출력
                        if tool_name == "tavily_search_results_json":
                            print(f"   # 검색 완료: {len(tool_output)}개 결과")
                            if tool_output:
                                first_result = tool_output[0].get('content', '')[:80]
                                print(f"     첫 결과: {first_result}...")
                        elif tool_name == "python_calculator":
                            print(f"   # 계산 결과: {tool_output}")
                        else:
                            print(f"   # 실행 완료")
                        
                        # 도구 결과를 메시지 히스토리에 추가
                        messages.append(
                            ToolMessage(
                                content=str(tool_output),
                                tool_call_id=tool_call['id']
                            )
                        )
                    except Exception as e:
                        print(f"   # 도구 실행 오류: {str(e)}")
                        messages.append(
                            ToolMessage(
                                content=f"오류 발생: {str(e)}",
                                tool_call_id=tool_call['id']
                            )
                        )
                else:
                    print(f"   # 알 수 없는 도구: {tool_name}")
            
            print(f"\n  💭 Agent가 다음 단계를 분석 중...")
            
        else:
            # Step 4: 도구 호출 없음 = 최종 답변
            print(" # Agent 판단: 충분한 정보 수집 및 계산 완료, 최종 답변 생성")
            print("\n" + "="*70)
            print(" # 최종 답변:")
            print("="*70)
            print(ai_msg.content)
            print("="*70)
            
            return
    
    # 최대 반복 횟수 도달
    print(f"\n  # 최대 반복 횟수({max_iterations})에 도달했습니다.")
    print("마지막 상태로 답변을 생성합니다.\n")
    
    final_msg = llm.invoke(messages + [
        HumanMessage(content="지금까지 수집하고 계산한 정보를 바탕으로 최종 답변을 작성해주세요.")
    ])
    
    print("="*70)
    print(" # 최종 답변 (강제):")
    print("="*70)
    print(final_msg.content)
    print("="*70)


if __name__ == "__main__":
    main()
