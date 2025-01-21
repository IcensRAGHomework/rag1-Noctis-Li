import json
import traceback
import requests
from pprint import pprint
import os

from model_configurations import get_model_configuration

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.prompts import PromptTemplate, MessagesPlaceholder, ChatPromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.tools import BaseTool, StructuredTool, tool
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

import base64
from mimetypes import guess_type

gpt_chat_version = 'gpt-4o'
gpt_config = get_model_configuration(gpt_chat_version)

calendarific_API = "4RPryAtPjbSbL3QdYKh4PphAAJJ9imkN"

llm = AzureChatOpenAI(
        model=gpt_config['model_name'],
        deployment_name=gpt_config['deployment_name'],
        openai_api_key=gpt_config['api_key'],
        openai_api_version=gpt_config['api_version'],
        azure_endpoint=gpt_config['api_base'],
        temperature=gpt_config['temperature']
    )

symbol_prompt = """
你是一個查詢工具,使用json格式返回用戶需求内容。
返回内容務必完整準確且僅返回用戶所需的内容。

回答範例：

{{
    "Result": [
        {{
            "date": "2024-10-10",
            "name": "國慶日"
        }}
    ]
}}

問題：{input}
"""

def regeneration_json(source):
    response = str(source).replace("```", "").replace("json", "")
    data = json.loads(response)
    return json.dumps(data, indent=4)

def generate_hw01(question):
    # pprint(symbol_prompt)

    prompt = PromptTemplate(input_variables=["input"], template=symbol_prompt)
    response = (prompt | llm).invoke({"input", question}).content
    return regeneration_json(source=response)

@tool
def get_holiday(country: str, year: int, month: int) -> str:
    '''獲得指定國家(兩位字母代碼形式)、年份、月份的節日信息(以json格式返回)'''
    params = {"country": country, "year": year, "month": month, "api_key": calendarific_API}
    url = "https://calendarific.com/api/v2/holidays"
    response = requests.get(url, params=params)
    print(response.url)
    ret = response.text
    print(ret)
    return ret

def generate_hw02(question):
    tool = [get_holiday]
    prompt = ChatPromptTemplate([
        ("system", """
你是一個查詢工具,使用json格式返回用戶需求内容。
返回内容務必完整準確且僅返回用戶所需的内容。

回答範例：

{{
    "Result": [
        {{
            "date": "2024-10-10",
            "name": "國慶日"
        }}
    ]
}}"""),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
        ("user", "{input}"),
    ])
    agent = create_openai_functions_agent(llm, tools=tool, prompt=prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tool, verbose=False)
    response = agent_executor.invoke({"input": question})
    return regeneration_json(source=response["output"])

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def generate_hw03(question2, question3):
    tool = [get_holiday]
    base_json = """
{{
    "Result": [
        {{
            "date": "2024-10-10",
            "name": "國慶日"
        }}
    ]
}}"""
    base_json2 = """
{{
    "Result": 
        {{
            "add": true,
            "reason": "蔣中正誕辰紀念日並未包含在十月的節日清單中。目前十月的現有節日包括國慶日、重陽節、華僑節、台灣光復節和萬聖節。因此，如果該日被認定為節日，應該將其新增至清單中。"
        }}
}}"""
    prompt = ChatPromptTemplate([
        ("system", """
你是一個查詢工具,使用json格式返回用戶需求内容。
返回内容務必完整準確且僅返回用戶所需的内容。

回答範例：
{template}"""),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
        ("user", "{input}"),
        ("ai", "{ai_response}"),
    ])
    agent = create_openai_functions_agent(llm, tools=tool, prompt=prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tool, verbose=False)
    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    response = agent_with_chat_history.invoke(
        {"template": base_json, "input":question2, "ai_response": generate_hw02(question2)},
        config={"configurable": {"session_id": "<foo>"}},
    )
    print("invoke 1:\n", response["output"])
    response = agent_with_chat_history.invoke(
        {"template": base_json2, "input":question3, "ai_response": "{output}"},
        config={"configurable": {"session_id": "<foo>"}},
    )
    print("invoke 2:\n", response["output"])
    return regeneration_json(response["output"])

# Function to encode a local image into data URL 
def local_image_to_data_url(image_path):
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'  # Default MIME type if none is found

    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"

def generate_hw04(question):
    image_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "baseball.png")
    print("image_path:", image_path)

    base_json = """
{{
    "Result": 
        {{
            "score": 5498
        }}
}}"""

    messages=[
        { "role": "system", "content": """
請用json格式回答用戶問題。
範例如下：
{{
    "Result": 
        {{
            "score": 5498
        }}
}}""" },
        { "role": "user", "content": [  
            { 
                "type": "text", 
                "text": question
            },
            { 
                "type": "image_url",
                "image_url": {
                    "url": local_image_to_data_url(image_path=image_path)
                }
            }
        ] } 
    ]
    response = llm.invoke(messages, max_tokens=2000)
    print(response)
    return regeneration_json(response.content)
    
def demo(question):
    llm = AzureChatOpenAI(
            model=gpt_config['model_name'],
            deployment_name=gpt_config['deployment_name'],
            openai_api_key=gpt_config['api_key'],
            openai_api_version=gpt_config['api_version'],
            azure_endpoint=gpt_config['api_base'],
            temperature=gpt_config['temperature']
    )
    message = HumanMessage(
            content=[
                {"type": "text", "text": question},
            ]
    )
    response = llm.invoke([message])
    
    return response
