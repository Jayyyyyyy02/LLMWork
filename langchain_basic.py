from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
# LLM 응답객체를 문자열로 변환해주는 파서

load_dotenv()

llm = ChatOpenAI(model = "gpt-4o-mini", temperature = 0.1)

# PromptTemplate을 이용하여 프롬프트를 작성해보자. 
# 문자열 템플릿을 기반으로 프롬프트를 자동으로 구성해준다
prompt = PromptTemplate.from_template("'{topic}' 주제에 대해 한 문장으로 설명해줘")
output_str = StrOutputParser()
# 응답 객체에서 텍스트(content)만 깔끔하게 추출해서 문자열로 변환해줌

# LCEL표현 (프롬프트 -> LLM모델 -> 출력) 파이프라인 연결
chain = prompt | llm | output_str

# 랭체인을 이용해서 실행해보자
response = chain.invoke({"topic":"LangChain"})
print(response)