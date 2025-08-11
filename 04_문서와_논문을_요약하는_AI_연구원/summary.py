from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path="../.env")
api_key = os.getenv('OPENAI_API_KEY')

def summarize_txt(file_path: str): # ①
    client = OpenAI(api_key=api_key)

    # ② 주어진 텍스트 파일을 읽어들인다.
    with open(file_path, 'r', encoding='utf-8') as f:
        txt = f.read()

    # ③ 요약을 위한 시스템 프롬프트를 생성한다.
    system_prompt = f'''
    너는 다음 글을 요약하는 봇이다. 아래 글을 읽고, 저자의 문제 인식과 주장을 파악하고, 주요 내용을 요약하라. 

    작성해야 하는 포맷은 다음과 같다. 
    
    # 제목

    ## 저자의 문제 인식 및 주장 (15문장 이내)
    
    ## 저자 소개

    
    =============== 이하 텍스트 ===============

    { txt }
    '''

    print(system_prompt)
    print('=========================================')

    # ④ OpenAI API를 사용하여 요약을 생성한다.
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.1,
        messages=[
            {"role": "system", "content": system_prompt},
        ]
    )

    return response.choices[0].message.content

if __name__ == '__main__':
    # file_path = './output/과정기반 작물모형을 이용한 웹 기반 밀 재배관리 의사결정 지원시스템 설계 및 구축_with_preprocessing.txt'
    file_path = './output/2303.11717v1_with_preprocessing.txt'    

    summary = summarize_txt(file_path)
    print(summary)

    # ⑤ 요약된 내용을 파일로 저장한다.
    with open('./output/arxiv_model_summary.txt', 'w', encoding='utf-8') as f:
        f.write(summary)