from dotenv import load_dotenv
from datetime import datetime
import pytz

from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
import os

import yfinance as yf
import streamlit as st
load_dotenv()

# 1. 툴 정의
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

@tool
def calculator(expression: str) -> str:
    '''
    간단한 산수 계산
        Args : expression 간단한 수식(예: 2 + 3*4)
    Returns:
        str : 수식 실행 결과값(예: 14)
    '''
    return str(eval(expression))

@tool
def get_stock_price(symbol: str) -> str:
    """
    주식 티커(symbol)를 입력받아 해당 종목의 최신 시세 정보와 기본 기업 정보를 조회해 문자열로 반환합니다.

    - 지원 예시:
        * 국내 KOSPI: "000660.KS", "005930.KS"
        * 국내 KOSDAQ: "293490.KQ"
        * 미국 주식: "AAPL", "GOOG"
    
    - 반환 정보:
        * 시가(Open), 고가(High), 저가(Low), 종가(Close)
        * 기업명(longName), 산업(industry), 섹터(sector)
        * 시가총액(marketCap), 공식 웹사이트 주소

    조회 실패 또는 존재하지 않는 티커일 경우 오류 메시지를 반환합니다.
    """
    print('주식 시세 가져올 예정')
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period = "1d")

        if data.empty:
            return f'{symbol} 정보가 없습니다'
        
        last = round(data['Close'].iloc[-1])
        open_price = round(data['Open'].iloc[-1])
        high = round(data['High'].iloc[-1])
        low = round(data['Low'].iloc[-1])
        info = stock.info

        name = info.get('l' \
        'ongName','정보 없음')
        sector = info.get('sector','정보 없음')
        industry = info.get('industry','정보 없음')
        website = info.get('website','정보 없음')
        market_cap = info.get('market_Cap','정보 없음')


        return f'''
            {symbol} ({name}) 시제 정보
            시가: {open_price}
            고가: {high}
            저가: {low}
            종가(현재가): {last}
            산업(industry): {industry}, 섹터: {sector}
            시가 총액: {market_cap}
            웹사이트: {website}
            '''
    except Exception as ex:
        return f'주식 정보 조회 오류: {ex}'
    
# ==티커 (심볼)========================================
# SK하이닉스   000660   KOSPI   000660.KS
# 삼성전자       005930   KOSPI   005930.KS
# 카카오       035720   KOSPI   035720.KS
# 애플          AAPL    나스닥(.O)
# 구글          GOOG    뉴욕증권거래소(.N)
# 카카오게임즈와 같은 코스닥 종목을 조회한다면 293490.KQ 를 사용 (코스닥은 KQ)
# =====================================================

# 2. llm 설정 -----------------------

llm = ChatOpenAI(model = "gpt-4o-mini", temperature = 0)
tools = [get_current_time, calculator, get_stock_price]

tool_dict = {t.name: t for t in tools}

# llm에 tools 바인딩
llm_with_tools = llm.bind_tools(tools)

# 3. streamlit UI 설정 ----------------------
st.set_page_config(page_title = 'Tools사용 AI Chatbot', page_icon ='🤖')
st.title("Langchain Tools 챗봇")
st.markdown("### 시간조회, 계산, 주식 시세/기업 정보를 질문하세요")

if "messages" not in st.session_state:
    st.session_state['messages'] = [
        SystemMessage("당신은 사용자 질문에 친절하게 답변하는 assistant입니다"
        + "\n사용자 질문에 답변하기 위해 tools를 사용할 수 있습니다")
    ]

def process_message(messages, user_input):
    messages.append(HumanMessage(user_input)) # system + user
    response = llm_with_tools.invoke(messages)
    messages.append(response)

    # tool 호출 결과 처리
    # for tool in response.tool_calls:
    for tool in getattr(response,'tool_calls', []):
        tool_name = tool['name']
        tool_args = tool.get('args', {})
        tool_func = tool_dict.get(tool_name)

        if tool_func:
            result = tool_func.invoke(tool_args)
            # 파이썬 함수 실행
            # ToolMessage객체 생성
            toolMsg = ToolMessage(
                content=result,
                tool_name = tool_name,
                tool_call_id = tool.get('id')
            )
            messages.append(toolMsg)
            return response
        
def print_chat_history():
    """
    스트림릿에 채팅 기록 출력하는 함수
    """
    for msg in st.session_state.messages[1:]: # SystemMessage는 제외
        if isinstance(msg, HumanMessage):
            st.markdown(f"**사용자: **{msg.content}")
        elif isinstance(msg, ToolMessage):
            st.markdown(f"**Tool 결과 (): **\n{msg.content}")
        else:
            st.markdown(f"**AI 응답: ** {msg.content}")

def print_chat_html():
    st.markdown("""
    <style>
    .chat-container {
        display: flex;
        flex-direction: column;
        gap: 10px;
    }

    .chat-bubble {
        padding: 10px 14px;
        border-radius: 12px;
        margin: 5px;
        max-width: 80%;
        line-height: 1.4;
        word-wrap: break-word;
        color: red;
    }

    .user {
        align-self: flex-end;
        background-color: yellow;
    }

    .ai {
        align-self: flex-start;
        background-color: skyblue;
    }

    .tool {
        align-self: flex-start;
        background-color: #E8F0FE;
        border-left: 4px solid #4285F4;
        font-size: 0.9em;
    }

    .label {
        font-weight: bold;
        margin-bottom: 4px;
        display: block;
    }
    </style>
    """, unsafe_allow_html = True)
    # 6. 챗 기록 출력
    st.markdown('<div class = "chat-container">', unsafe_allow_html = True)

    for msg in st.session_state.messages[1:]:  # SystemMessage 제외
        if isinstance(msg, HumanMessage):
            st.markdown(f"""
            <div class = "chat-bubble user">
                <span class = "label">사용자</span>
                {msg.content}
            </div>
            """, unsafe_allow_html = True)

        elif isinstance(msg, ToolMessage):
            st.markdown(f"""
            <div class = "chat-bubble tool">
                <span class = "label">🔧 툴 결과 ({msg.tool_name})</span>
                <pre>{msg.content}</pre>
            </div>
            """, unsafe_allow_html = True)

        else:
            st.markdown(f"""
            <div class = "chat-bubble ai">
                <span class = "label">AI</span>
                {msg.content}
            </div>
            """, unsafe_allow_html = True)

    st.markdown('</div>', unsafe_allow_html = True)


# 사용자 입력
user_input = st.chat_input("질문 입력: ")

if user_input:
    response = process_message(st.session_state.messages, user_input)

# 채팅 기록 출력
# print_chat_history()
print_chat_html()