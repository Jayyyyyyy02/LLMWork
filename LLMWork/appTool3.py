from dotenv import load_dotenv
from datetime import datetime
import pytz
import os

from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_community.tools.tavily_search import TavilySearchResults

import yfinance as yf
import streamlit as st

# .env 파일에서 API 키 로드 (OPENAI_API_KEY, TAVILY_API_KEY)
load_dotenv()

# 1. 툴 정의 --------------------------------------------------

@tool
def get_current_time(timezone: str, location: str) -> str:
    '''
    현재 시간을 알려주는 tool입니다.
    Args:
        timezone(str): 타임존(예: 'Asia/Seoul')
        location(str): 지역명(예: 서울)
    '''
    try:
        tz = pytz.timezone(timezone)
        now = datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S')
        return f'{timezone} ({location}) 현재 시각: {now}'
    except Exception as e:
        return f'오류: {e}'

@tool
def calculator(expression: str) -> str:
    '''간단한 산수 계산을 수행합니다. (예: 2 + 3 * 4)'''
    try:
        return str(eval(expression))
    except Exception as e:
        return f"계산 오류: {e}"

@tool
def get_stock_price(symbol: str) -> str:
    """주식 티커(symbol)를 입력받아 실시간 시세 및 기업 정보를 조회합니다."""
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period="1d")
        if data.empty: return f'{symbol} 정보가 없습니다.'
        
        info = stock.info
        last = round(data['Close'].iloc[-1], 2)
        return f"{symbol} 현재가: {last}, 섹터: {info.get('sector')}, 웹사이트: {info.get('website')}"
    except Exception as ex:
        return f'주식 정보 조회 오류: {ex}'

# Tavily 검색 툴 생성 (최대 3개의 결과 반환)
tavily_search = TavilySearchResults(k=3)

# 2. LLM 및 도구 바인딩 -----------------------------------------

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 모든 툴을 리스트에 담기 (Tavily 포함)
tools = [get_current_time, calculator, get_stock_price, tavily_search]

# LLM이 인식할 수 있도록 이름 매핑 사전 생성
tool_dict = {
    "get_current_time": get_current_time,
    "calculator": calculator,
    "get_stock_price": get_stock_price,
    "tavily_search_results_json": tavily_search # LangChain 기본 제공 이름
}

llm_with_tools = llm.bind_tools(tools)

# 3. Streamlit UI 설정 ----------------------------------------

st.set_page_config(page_title='Advanced Tools Chatbot', page_icon='🔍')
st.title("LangChain Multi-Tool 챗봇")
st.markdown("현재 시간, 계산기, 주식 정보, 그리고 **실시간 웹 검색**이 가능합니다.")

if "messages" not in st.session_state:
    st.session_state['messages'] = [
        SystemMessage(content="당신은 도구를 적절히 활용하여 답변하는 유능한 비서입니다. 최신 정보가 필요하면 tavily_search_results_json을 사용하세요.")
    ]

def process_message(messages, user_input):
    messages.append(HumanMessage(content=user_input))
    
    # 1차 호출: LLM이 도구 사용 여부 결정
    response = llm_with_tools.invoke(messages)
    messages.append(response)

    # 도구 호출이 발생한 경우 처리
    if response.tool_calls:
        for tool_call in response.tool_calls:
            t_name = tool_call['name']
            t_args = tool_call['args']
            
            # 검색 툴의 경우 이름이 다를 수 있으므로 체크
            actual_tool = tool_dict.get(t_name)
            
            if actual_tool:
                # 도구 실행
                result = actual_tool.invoke(t_args)
                # 도구 결과 메시지 추가
                tool_msg = ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call['id'],
                    name=t_name
                )
                messages.append(tool_msg)
        
        # 도구 결과를 바탕으로 최종 답변 생성
        final_response = llm_with_tools.invoke(messages)
        messages.append(final_response)

# --- 채팅 내역 출력 CSS 및 함수 (제공해주신 HTML 스타일 유지) ---
def print_chat_html():
    st.markdown("""
    <style>
    .chat-container { display: flex; flex-direction: column; gap: 10px; }
    .chat-bubble { padding: 10px 14px; border-radius: 12px; margin: 5px; max-width: 80%; line-height: 1.4; }
    .user { align-self: flex-end; background-color: #FFF9C4; color: black; } /* 노란색 계열 */
    .ai { align-self: flex-start; background-color: #E1F5FE; color: black; } /* 하늘색 계열 */
    .tool { align-self: flex-start; background-color: #F5F5F5; border-left: 4px solid #9E9E9E; font-size: 0.85em; color: #555; }
    .label { font-weight: bold; margin-bottom: 4px; display: block; font-size: 0.8em; }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for msg in st.session_state.messages:
        if isinstance(msg, SystemMessage): continue
        
        if isinstance(msg, HumanMessage):
            cls, label = "user", "사용자"
        elif isinstance(msg, ToolMessage):
            cls, label = "tool", f"🔧 Tool: {getattr(msg, 'name', '함수')}"
        else:
            cls, label = "ai", "AI 응답"
            if not msg.content: continue # Tool 호출만 있는 메시지는 패스

        st.markdown(f"""
        <div class="chat-bubble {cls}">
            <span class="label">{label}</span>
            {msg.content}
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# 사용자 입력 처리
user_input = st.chat_input("무엇이든 물어보세요! (예: 오늘 삼성전자 주가랑 최근 뉴스 알려줘)")

if user_input:
    process_message(st.session_state.messages, user_input)
    st.rerun()

print_chat_html()