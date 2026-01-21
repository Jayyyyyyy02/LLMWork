# ragChat.py
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    print('api_key없음')
    raise ValueError('api key 없음') #예외 발생


embedding = OpenAIEmbeddings(api_key=api_key,
                            model='text-embedding-3-large')
# print('embedding: ',embedding)\

# 크로마 디비
from langchain_chroma import Chroma
persist_directory = './chroma_store'
#처음 만들 때는 Chroma.from_documents(...,embedding,....)
#기존 만들어진 크로마 로딩시에는 Chroma(...,embedding_function,...)
vector_store = Chroma(
    persist_directory = persist_directory,
    embedding_function = embedding
)
print('# 벡터 스토아 로딩 성공###')

# llm 언어 모델
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model = "gpt-4o-mini", api_key = api_key)

# 도큐먼트 체인 생성
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate

template="사용자 질문에 대해 context에 기반해서 답변하세요\n\n{context}"
qna_prompt = ChatPromptTemplate([
    SystemMessagePromptTemplate.from_template(template),
    MessagesPlaceholder(variable_name = "messages") #대화 히스토리 포함
])

document_chain = create_stuff_documents_chain(llm,qna_prompt)
#문서 조각을 하나로 합쳐서 llm의 context에 집어넣고(채우고) 결과를 생성=>문서체인을 만든다

query_augumentation_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("""
        너는 질문 보정 전문 AI야.
        이전 대화를 참고해 모호한 질문을 명확히 바꾸는 게 목적이야.
        대명사나 이, 저, 그와 같은 표현을 명확한 명사로 표현해
        이전 대화 맥락과 상관없이, 새로운 질문에 대한 대상 도시나 주제를 명확하게 파악하고 보정해. 
        보정된 질문이 원래 질문의 의도와 다르게 해석되지 않도록 주의해
                                            
        **절대로 질문에 대한 답변을 하지 말고, 보정된 질문 문장 하나만 출력해.** 
        예시: 서울의 녹지 공간 확대 계획은 무엇인가요?                                              
        """),    
    #MessagesPlaceholder(variable_name="messages"),
    # 과거 대화 삽입 (이전 대화가 messages변수에 들어가서 대화의 맥락을 구성)

    HumanMessagePromptTemplate.from_template("{query}")
    # 새로운 질문 (마지막으로 {query}에 새로운 질문으로 들어감)
])


query_augument_chain = query_augumentation_prompt|llm
#llm이 모호한 질문을 명확하게 바꿔준다

#리트리버 (검색)
retriever = vector_store.as_retriever(k=3)
#k=3: 코사인 유사도를 이용해서 유사한 문서 조각 3개를 가져오도록 설정

# retriever: 임베딩 기반으로 관련 문서 k개 찾기               
# query_augument_chain : 모호한 질문을 명확하고 검색에 적합한 형태로 보정(증강)
# document_chain: 검색된 문서를 context로 활용해 LLM이 최종 답변 생성 

# 채팅 UI 구현
import streamlit as st
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage

st.header("::🐕LangChain Chatbot with RAG::")

if "messages" not in st.session_state:
    st.session_state['messages'] = [
        SystemMessage("너는 문서에 기반하여 답변하는 도시정책 전문가야")
    ]

# 화면에 메시지 출력
for msg in st.session_state.messages:
    if isinstance(msg, SystemMessage):
        who = "system"
    elif isinstance(msg, AIMessage):
        who = "assistant"
    else:
        who = "user"
    st.chat_message(who).write(msg.content)

def get_ai_response(messages, docs):
    response = document_chain.stream({
        "messages":messages,
        "context":docs
    })
    #RAG기반으로 답변을 얻어옴

    #전체답변을 한 번에 만들지 않고, 조금씩 chunk로 순차적으로 내보내다=>스트리밍
    for chunk in response:
        yield chunk 
        #yield chunk => 응답 조각을 순차적으로 내보낸다


#사용자 입력받기
if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    st.session_state.messages.append(HumanMessage(prompt))
    print("User: ", prompt)

    #사용자 입력한 질문을 이용해 확장된 질의를 만들자
    augmented_query = query_augument_chain.invoke({
        #"messages": st.session_state["messages"],
        "query":prompt
    })
    print("augmented_query: ",augmented_query)
    st.info(f"검색용 질의문 : {augmented_query}", icon="💡")

    print("="*70)
    print("관련 문서 검색")
    print("="*70)
    docs = retriever.invoke(f"{prompt}\n{augmented_query}")
    #벡터 디비에서 관련 문서 가져옴

    for doc in docs:
        print(doc)
        print('-'*70)
        with st.expander(f"문서: {doc.metadata.get('source',"알수 없음")}"):
            #파일명과 페이지 정보 출력
            st.write(f"page:{doc.metadata.get('page','')}")
            st.write(doc.page_content)
    print("="*70)

    #AI 답변 출력
    with st.spinner(f"AI가 답변을 준비 중입니다...{augmented_query}"):
        response = get_ai_response(st.session_state.messages, docs)
        result = st.chat_message('assistant').write_stream(response)
                #응답을 스트리밍 방식으로 출력
        st.session_state['messages'].append(AIMessage(result))