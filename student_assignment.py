import json
import traceback
from pprint import pprint

from model_configurations import get_model_configuration

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
from langchain.prompts import PromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage

gpt_chat_version = 'gpt-4o'
gpt_config = get_model_configuration(gpt_chat_version)

def generate_hw01(question):
    # Initialize AzureChatOpenAI model
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

    # pprint(symbol_prompt)

    prompt = PromptTemplate(input_variables=["input"], template=symbol_prompt)
    response = (prompt | llm).invoke({"input", question}).content
    print("response:\n",response)
    response = str(response).replace("```", "").replace("json", "")
    data = json.loads(response)
    print("data:\n", data)
    ret = json.dumps(data, indent=4)
    return ret
    # pprint(llm.invoke(prompt.from_messages(input=question)))
    
def generate_hw02(question):
    pass
    
def generate_hw03(question2, question3):
    pass
    
def generate_hw04(question):
    pass
    
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
