import yfinance as yf
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
if __name__ == '__main__':
    result = get_stock_price('005930.KS')
    print(result)