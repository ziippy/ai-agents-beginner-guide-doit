import os
from dotenv import load_dotenv
load_dotenv(dotenv_path="../.env")

from langchain_openai import ChatOpenAI

# 모델 초기화
model = ChatOpenAI(model="gpt-4o-mini")

# 상태 정의하기
from typing import Annotated # annotated는 타입 힌트를 사용할 때 사용하는 함수
from typing_extensions import TypedDict # TypedDict는 딕셔너리 타입을 정의할 때 사용하는 함수

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

class State(TypedDict):	# State 클래스는 TypedDict를 사용하여 딕셔너리 형태로 관리
    """
    State 클래스는 TypedDict를 상속받습니다.

    속성:
        messages (Annotated[list[str], add_messages]): 메시지들은 "list" 타입을 가집니다.   # messages라는 변수만 포함되며, 이는 Annoted를 사용해 문자열로 구성된 리스트 형식임
       'add_messages' 함수는 이 상태 키가 어떻게 업데이트되어야 하는지를 정의합니다.  # add_messages 함수를 추가. 이는 langgraph에서 제공하는 함수로, 문자열이 주어질 때 이를 추가하는 기능 수행
        (이 경우, 메시지를 덮어쓰는 대신 리스트에 추가합니다)
    """
    messages: Annotated[list[str], add_messages]

# StateGraph 클래스를 사용하여 State 타입의 그래프를 생성합니다.
graph_builder = StateGraph(State)   # 생성한 State 를 이용해 StateGraph를 만들어 graph_builder라는 변수에 담는다.

# 노드 생성하기
def generate(state: State):
    """
    주어진 상태를 기반으로 챗봇의 응답 메시지를 생성합니다.

    매개변수:
    state (State): 현재 대화 상태를 나타내는 객체로, 이전 메시지들이 포함되어 있습니다.
		
    반환값:
    dict: 모델이 생성한 응답 메시지를 포함하는 딕셔너리. 
          형식은 {"messages": [응답 메시지]}입니다.
    """ 
    return {"messages": [model.invoke(state["messages"])]}

graph_builder.add_node("generate", generate)

# 엣지 설정하기
graph_builder.add_edge(START, "generate")
graph_builder.add_edge("generate", END)

####### 메모리 추가
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()

graph = graph_builder.compile(checkpointer=memory)
# graph = graph_builder.compile()

config = {"configurable": {"thread_id": "abcd"}}

##############################

from langchain.schema import HumanMessage

while True:
    user_input = input("You\t: ")

    if user_input in ["exit", "quit", "q"]:
        break

    for event in graph.stream({"messages": [HumanMessage(user_input)]}, config, stream_mode="values"):
    # for event in graph.stream({"messages": [HumanMessage(user_input)]}, stream_mode="values"):
        event['messages'][-1].pretty_print()  # 마지막 메시지를 예쁘게 출력합니다.

    print(f"\n현재 메시지 개수: {len(event['messages'])}\n----------------------------------------\n")