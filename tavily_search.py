'''
- 타빌리 api를 이용해서 최신 소식을 검색해보자
'''
import os
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv

load_dotenv()
tavily_api_key = os.getenv('TAVILY_API_KEY')
print(tavily_api_key[:8])
if not tavily_api_key:
    print('tavily api key를 찾을 수 없어요')

def web_search():
    search_tool = TavilySearchResults(
        max_results = 3, # 최대 3개 검색결과 반환
        api_key = tavily_api_key
    )
    # 기본 내장 툴: tavily_search_results_json => 내장 툴 이름(이미 정의된 이름)
    print(search_tool.name)

    question = "2025년 롤드컵 우승자를 알려줘" 

    print(f'질문: {question}')
    print('Tavily 검색 중...')

    # 검색 실행
    search_result = search_tool.invoke({"query": question})

    print('='*100)
    print('검색 결과: ')
    for i, result in enumerate(search_result, 1):
        content = result.get('content','')[:200]
        source = result.get('url', '')
        print(f"{i} : {content}")
        print(f"출처: {source}")




if __name__ == '__main__':
    web_search()