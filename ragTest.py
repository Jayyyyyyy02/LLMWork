# ragTest.py
from langchain_community.document_loaders import PyPDFLoader

#loader = PyPDFLoader("./data/2040_seoul_plan.pdf")
#data_seoul = loader.load()
#문서를 로드 (pdf전체 읽어서 페이지 별로 나누고 메모리에 로드함)
#print(len(data_seoul), type(data_seoul)) #205 <class 'list'>

# 5~7페이지 문서를 출력해보자
#for doc in data_seoul[5:8]:
#    print(doc.page_content[:500]) #각 페이지의 500자만 출력
#    print('='*100)
'''
매번 실행할 때마다 로드할 수 없으므로 pickle을 이용해서 저장
(데이터를 파일로 직렬화해서 저장) pickle은 객체를 그대로 파일로 직렬화해서 
저장하고 다시 객체로 복원(역직렬화)하는 도구
pickle을 이용하면 한 번 로드한 데이터를 저장했다가 다음 실행할 때
바로 불러올 수 있어서 빠르다
'''
import pickle
import os

if os.path.exists('./data/data_seoul.pkl'):
    #해당 파일이 있으면 로드하기 => 빠름
    with open('./data/data_seoul.pkl','rb') as f: #읽기 모드/바이너리 모드
        data_seoul = pickle.load(f)
else:
    #pkl파일 없으면 pdf를 로드하고 pkl파일로 저장
    loader = PyPDFLoader('./data/2040_seoul_plan.pdf')
    data_seoul = loader.load()
    with open('./data/data_seoul.pkl','wb') as f: #쓰기 모드/바이너리 모드
        pickle.dump(data_seoul, f)

# 페이지 일부만 확인해보자
if os.path.exists('./data/data_seoul.pkl'):
    print(data_seoul[10].page_content[:500])
else:
    print('./data/data_seoul.pkl 파일이 없어요!! 종료합니다')
    exit(0)

# 여러 문서를 읽어야 하므로 함수로 모듈화하여 재사용하자
def load_pdf_with_pickle(pdf_path, pickle_path = None):
    '''
    pdf파일 로드하고 pickle로 캐싱하는 함수
    - pdf_path: pdf파일 경로
    - pickle_path : pickle파일 경로 (지정하지 않으면 pdf이름 기반)
    '''
    if pickle_path is None:
        pickle_path = pdf_path.replace('.pdf','.pkl')
        #pkl파일이 없으면 pdf파일명에서 .pdf를 .pkl로 대체해서 피클 경로 만들자

    if os.path.exists(pickle_path):
        with open(pickle_path,'rb') as f:
            data=pickle.load(f)
        print(f'Loaded from pickle: {pickle_path}')
    else:
        #피클 파일이 없다면 pdf 로드후 피클로 저장
        loader= PyPDFLoader(pdf_path)
        data = loader.load()
        with open(pickle_path,'wb') as f:
            pickle.dump(data, f)
        print(f'Loaded from PDF and Saved pickle: {pickle_path}')
    return data

# 뉴욕문서 읽어보자
data_nyc = load_pdf_with_pickle('./data/OneNYC_2050_Strategic_Plan.pdf')
#data_seoul=load_pdf_with_pickle('./data/2040_seoul_plan.pdf')
print("*"*70)
data_nyc[3].page_content[:500]

'''
PyPDFLoader의 load()가 반환하는 데이터는 문서 리스트
[
    Document(page_content="첫 번째 페이지 텍스트", metadata={...}),
    Document(page_content="두 번째 페이지 텍스트", metadata={...}),
    ...
]
'''
# 문서 로드 끝=================================
# RecursiveCharacterTestSplitter를 이용해서
# 텍스트를 청크로 나누자
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, 
                                            chunk_overlap = 100)
# 텍스트 데이터를 1000자씩 나누고 100자의 오버랩 설정
# 오버랩을 적절히 사용하면 맥락이 이어지고 중요한 정보가 잘려서 사라지지 않도록
# 할 수 있음

# 청크 출력하기-------------
all_splits = text_splitter.split_documents(data_seoul)

for i, split in enumerate(all_splits):
    print(f'Split {i+1} :------------------')
    # print(split)

print(type(all_splits[0]))
print(all_splits[50].page_content)
print('*'*100)
print(all_splits[51].page_content)

# 오버랩 설정을 해도 문서에 따라 중복되지 않는 경우가 있다
# 이런 경우 오버랩되도록 로직을 구성해야 한다
for i in range(len(all_splits)-1):
    all_splits[i].page_content +="\n"+all_splits[i+1].page_content[:100]
    # i번째 내용에 i+1번째 내용의 100자를 누적시키기
print('#'*100)
print(all_splits[50].page_content)
print('#'*100)
print(all_splits[51].page_content)

# 뉴욕 데이터도 청킹 하자
ny_splits = text_splitter.split_documents(data_nyc)
for i,split in enumerate(ny_splits):
    print(f'NY Split {i+1}--------------')
print('*'*100)

print(ny_splits[50].page_content)
print('*'*100)
print(ny_splits[51].page_content)

#서울과 뉴욕 청크 문서를 하나로 합쳐 리스트로 
# 만들자(벡터DB에 임베딩해서 넣기 위해)
print("합치기 전 크기: ",len(all_splits)) #합치기 전 크기:  308
all_splits.extend(ny_splits)
print("합친 후  크기: ",len(all_splits)) #합친 후  크기:  1331
# 서울문서 청크길이 + 뉴욕문서 청크길이
print("===OpenAIEmbeddings 이용해 임베딩하기==========")
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_chroma import Chroma

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
embedding = OpenAIEmbeddings(
    model='text-embedding-3-large',api_key=api_key
)
#문서를 임베딩처리하고 embedding 을 벡터DB에 전달하면 =>벡터DB에 임베딩 데이터 저장
#테스트 차원에서 질의문을 임베딩해보자
vector = embedding.embed_query("뉴욕의 온실가스 저장 정책에 대해 알려줘")
print(len(vector)) #3072
print(vector[:10])

# 벡터DB에 all_splits를 임베딩하여 저장하자########
persist_directory = './chroma_store'

if not os.path.exists(persist_directory):
    print('Creating new Chroma store')
    #처음 생성시 =Chroma.from_documents() 함수 사용
    vector_store = Chroma.from_documents(documents = all_splits,
                    embedding = embedding,
                    persist_directory = persist_directory)
                    # 문서-> 임베딩 생성+저장 
else:
    print('Loading existing Chroma store')
    #크로마 스토어가 있을 경우 Chroma()생성자 호출
    #embedding_function 에 임베딩모델 할당
    vector_store = Chroma(persist_directory = persist_directory,
                        embedding_function = embedding)
                        #검색용

# 문서-> 임베딩 모델(벡터 생성)->크로마DB 저장(벡터 저장)
# 질문-> 임베딩 모델(벡터 생성)->크로마DB(KNN 검색-코사인 유사도 사용)
# -> 유사한 청크를 반환

# 주어진 청크 기반 언어모델로 답변하는 시스템 만들기=====
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model = "gpt-4o-mini", temperature = 0)

qna_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "사용자 질문에 대해 아래 context에 기반하여 답변해\n\n{context}"
    ),
    MessagesPlaceholder(variable_name = "messages") 
    #chat_history message들어감
])
#검색된 내용을  context에 포함할 예정

document_chain = create_stuff_documents_chain(llm, qna_prompt)
#여러 문서를 하나로 합쳐서 LLM에 넣는 "stuff" 체인전략을 만들어주는 함수

from langchain_community.chat_message_histories import ChatMessageHistory
chat_history = ChatMessageHistory()
chat_history.add_user_message("서울시의 온실가스 저감 정책에 대해 알려줘")

retriever = vector_store.as_retriever(k=3)
#유사한 것 3개 가져오도록 지정 k=3 => 유사한 청크수
docs = retriever.invoke("서울시의 온실가스 저감 정책에 대해 알려줘")
#벡터 db에서 검색한 결과 받아서 context로 활용한다

answer = document_chain.invoke({
    "messages": chat_history.messages,
    "context": docs
})
chat_history.add_ai_message(answer)
print(answer)
print("*"*70)
for msg in chat_history.messages:
    print(msg)
    print('-'*70)

q1="뉴욕은?"
# 분명하지 않은 질문을 질의 확장 체인을 실행하여 명확히 해보자
#=>증강(augumentation)

template = '''
너는 질문 보정 전문 AI야. 이전 대화를 참고해서 모호한 질문을 명확히
바꾸는게 목적이야
'''

q_augument_prompt = ChatPromptTemplate.from_messages([
    ("system",template),
    MessagesPlaceholder(variable_name="messages"),
    ("human","{query}")
])
from langchain_core.output_parsers import StrOutputParser

q_augument_chain = q_augument_prompt|llm|StrOutputParser()
augument_query = q_augument_chain.invoke({
    "messages": chat_history.messages,
    "query":q1
})
print("증강된 쿼리문: ", augument_query)

#증강된 쿼리로 리트리버로 다시 검색 실행
docs = retriever.invoke(augument_query)
#augument_query=>임베딩 되어 벡터로 변환 => 크로마디비에서 코사인 유사도로 비교
# => 가장 유사한 문서 top-k개를 검색해서 반환

for doc in docs:
    print(doc)
    print('='*70)

chat_history.add_user_message(q1)

# Generation
#증강된 질문 + 검색된 문서로 답변 생성
answer = document_chain.invoke({
    "messages":chat_history.messages,
    "context": docs
})
chat_history.add_ai_message(answer) #모델 응답 저장
print("#"*70)
print(answer)
#LLM은 증강된 질의와 새로 검색된 문서를 기반으로 답변을 생성한다
#원문=>영어, 결과는??