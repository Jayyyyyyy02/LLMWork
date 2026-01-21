loader = PyPDFLoader("./data/2040_seoul_plan.pdf")
data_seoul = loader.load()
# 문서를 로드 (pdf전체 읽어서 페이지 별로 나누고 메모리에 로드함)
print(len(data_seoul), type(data_seoul))

# 5 ~ 7 페이지 문서를 출력해보자
for doc in data_seoul[5:8]:
    print(doc.page_content[:500]) # 각 페이지의 500자만 출력
    print('='*100)