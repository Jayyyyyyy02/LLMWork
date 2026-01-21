import streamlit as st

# st.header() : #h2
st.subheader(":::메아리(Echo) 채팅:::") #h3

# 스트림릿은 사용자 액션이 발생할 때마다 스크립트를 위에서부터 다시 실행한다
# 이렇게 되면 대화내용이 초기화된다. 이 대화내용 상태를 보관해야 채팅 구현 가능하다
# st.session_state => 메시지 저장공간
# session_state에 messages라는 키값으로 대화내용을 저장해보자
# => 이렇게 저장하지 않으면 이전 대화가 사라진다
if "messages" not in st.session_state:
    st.session_state['messages'] = [] 
    # role, content 기반 메시지 저장 공간

# 기존에 저장된 메시지가 있다면 화면에 출력
for msg in st.session_state['messages']:
    with st.chat_message(msg['role']):
        st.write(msg["content"])

# 사용자 대화내용 입력받기
user_input = st.chat_input("메시지를 입력하세요")

if user_input:
    # 사용자 메시지를 session_state에 저장+화면에 출력
    st.session_state['messages'].append({
        "role":"user",
        "content":user_input
    })
    # 챗봇 응답 생성 (임의의 답변을 만들어서 session_state['messages']에 저장하세요)
    # role: 'assistant', content:'내용'
    aibot = f"Echo>> {user_input}"

    st.session_state['messages'].append({
        "role":'assistant',
        "content": aibot
    })

    st.rerun() 
    #입력후 내용이 바로 화면에 반영되지 않음. 이럴 때 rerun()을 호출하면 화면에 바로 출력된다