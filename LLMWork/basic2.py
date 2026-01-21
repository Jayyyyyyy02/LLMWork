from openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv() #환경설정 파일(.env)안에 기술된 환경변수들을 메모리에 올린다
api_key = os.getenv('OPENAI_API_KEY')
print(f'API KEY: {api_key[:5]} ... {api_key[-5:]}')
client=OpenAI(api_key=api_key)

response = client.chat.completions.create(model='gpt-4o-mini', temperature=0.1,messages=[
    {"role":"system", "content":"당신은 유용한 도움을 주는 assistant입니다"},
    {"role":"user","content":"2025년 챔피언십(롤드컵) 결승전 우승자는 누구야?"}
])
# print(response)
print('*'*100)
print(response.choices[0].message.content)
print('*'*100)
print(f'모델: {response.model}')
print(f'사용한 토큰: {response.usage.total_tokens}')