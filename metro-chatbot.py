import os
from dotenv import load_dotenv
from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Fetch API Key from .env config
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OpenAI API Key 未設定")

# 定義對話狀態
class ChatState(TypedDict):
    messages: list[AnyMessage]
    intent: str
    is_metro_related: bool

class MetroChatbot:
    def __init__(self):
        # init LLM
        self.llm = ChatOpenAI(model="gpt-4o")
        
        # Prompt of intent classifiy 
        self.intent_prompt = ChatPromptTemplate.from_messages([
            ("system", """
            你是一個專業的捷運客服意圖分類助手。請根據用戶的問題判斷其意圖和是否與捷運相關。
            捷運相關的問題類別包括：
            1. 售票問題
            2. 路線問題
            3. 捷運站設施問題
            4. 營運時間問題

            請按以下格式回覆：
            is_metro_related: [true/false]
            intent: [intent_category]
            
            如果不是捷運相關問題，則 is_metro_related 為 false，intent 可以是 "不相關"
            """),
            ("human", "{input}")
        ])
        
        # Create a workflow
        self.workflow = StateGraph(ChatState)
        
        # add nodes
        self.workflow.add_node("classify_intent", self.classify_intent)
        self.workflow.add_node("handle_metro_query", self.handle_metro_query)
        self.workflow.add_node("handle_non_metro_query", self.handle_non_metro_query)
        
        # add condition edge
        self.workflow.add_conditional_edges(
            "classify_intent",
            self.route_intent,
            {
                "metro_query": "handle_metro_query",
                "non_metro_query": "handle_non_metro_query"
            }
        )
        
        # End
        self.workflow.add_edge("handle_metro_query", END)
        self.workflow.add_edge("handle_non_metro_query", END)
        
        # Set entry
        self.workflow.set_entry_point("classify_intent")
        
        # Compile
        self.app = self.workflow.compile()
    
    def classify_intent(self, state: ChatState):
        """classify_intent"""
        # Get last message
        last_message = state['messages'][-1]
        
        # Fetch content via different type
        if isinstance(last_message, dict):
            content = last_message.get('content', '')
        elif hasattr(last_message, 'content'):
            content = last_message.content
        else:
            content = str(last_message)
        
        intent_chain = self.intent_prompt | self.llm | StrOutputParser()

        # get retuls from LLM
        result = intent_chain.invoke({"input": content})
        
        is_metro_related = "true" in result.lower()
        intent = result.split("intent: ")[-1].strip() if is_metro_related else "不相關"
        
        return {
            **state, 
            "intent": intent,
            "is_metro_related": is_metro_related
        }

    def route_intent(self, state: ChatState):
        """route_intent"""
        return "metro_query" if state["is_metro_related"] else "non_metro_query"
    
    def handle_metro_query(self, state: ChatState):
        """handle_metro_query"""
        intent = state["intent"]
        # last_mesage = state['messages'][-1].contents
        
        # todo call differenct tools by topics.
        responses = {
            "售票問題": "關於售票的詳細資訊...",
            "路線問題": "捷運路線規劃建議...",
            "捷運站設施問題": "捷運站設施相關說明...",
            "營運時間問題": "捷運營運時間詳細資訊..."
        }
        
        response = responses.get(intent, "很抱歉，我無法明確回答您的問題")
        
        return {
            **state,
            "messages": state["messages"] + [
                {"role": "assistant", "content": response}
            ]
        }
    
    def handle_non_metro_query(self, state: ChatState):
        """handle_non_metro_query"""
        response = "對不起，這個問題似乎不是關於捷運的。我只能回答與捷運相關的問題。"
        
        return {
            **state,
            "messages": state["messages"] + [
                {"role": "assistant", "content": response}
            ]
        }
    
    def run(self, query):
        """run Agents"""
        initial_state = {
            "messages": [{"role": "human", "content": query}],
            "intent": "",
            "is_metro_related": False
        }
        
        result = self.app.invoke(initial_state)
        return result["messages"][-1].get("content", result["messages"][-1])

# 使用範例
def main():
    chatbot = MetroChatbot()
    
    # 測試不同類型的查詢
    queries = [
        "請問捷運票價是多少？",
        "我想知道從A站到B站要怎麼走",
        "捷運站裡有無廁所？",
        "今天天氣不錯",
        "捷運幾點開始營運？",
        "可以幫我寫一段河內塔的C++ code 嗎？"
    ]
    
    for query in queries:
        print(f"Query: {query}")
        response = chatbot.run(query)
        print(f"Response: {response}\n")

if __name__ == "__main__":
    main()
