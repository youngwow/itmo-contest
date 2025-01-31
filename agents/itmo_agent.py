from langchain_gigachat.chat_models import GigaChat
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import END, StateGraph
from typing import TypedDict, List, Optional
import os
from pydantic import HttpUrl
import re

class AgentState(TypedDict):
    query: str
    id: int
    options: Optional[List[str]]
    llm_answer: Optional[str]
    search_results: Optional[List[dict]]
    final_answer: Optional[dict]

def parse_query(state: AgentState) -> AgentState:
    query = state["query"]
    options = re.findall(r"\n(\d+)\..+", query)
    state["options"] = options if len(options) > 1 else None
    return state

def generate_initial_answer(state: AgentState) -> AgentState:
    llm = GigaChat(
        credentials=os.environ["GIGACHAT_CREDENTIALS"],
        verify_ssl_certs=False)
    prompt = f"""
    Ответь на вопрос о Университете ИТМО. 
    Вопрос: {state['query']}
    Ответ должен быть кратким и фактологически точным.
    """
    state["llm_answer"] = llm.invoke(prompt).content
    return state

def search_external(state: AgentState) -> AgentState:
    search = TavilySearchResults(
        max_results=3,
        include_answer=True,
        include_raw_content=False,
        include_images=False,
        include_domains=['itmo.ru'])
    query = state["query"]
    results = search.invoke({"query": query})
    state["search_results"] = [res for res in results[:3]] 
    return state

def decide_answer(state: AgentState) -> AgentState:
    sources = []
    reasoning = []
    
    if state["llm_answer"]:
        reasoning.append(f"Согласно языковой модели GigaChat: {state['llm_answer']}")
    
    if state["search_results"]:
        sources = [HttpUrl(res["url"]) for res in state["search_results"]]
        reasoning.append("Дополнительная информация из внешних источников.")
    
    answer = None
    if state["options"]:
        options_text = "\n".join(state["options"])
        prompt = f"""
        На основе информации определи правильный вариант:
        {state['llm_answer']}
        Варианты:
        {options_text}
        Ответ только цифрой.
        """
        llm = GigaChat(
            credentials=os.environ["GIGACHAT_CREDENTIALS"],
            verify_ssl_certs=False,
            temperature=0)
        answer = llm.invoke(prompt).content.strip()
    
    state["final_answer"] = {
        "id": state["id"],
        "answer": int(answer) if answer and answer.isdigit() else None,
        "reasoning": " ".join(reasoning),
        "sources": sources
    }
    return state

def create_agent():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("parse_query", parse_query)
    workflow.add_node("generate_initial_answer", generate_initial_answer)
    workflow.add_node("search_external", search_external)
    workflow.add_node("decide_answer", decide_answer)
    
    workflow.set_entry_point("parse_query")
    workflow.add_edge("parse_query", "generate_initial_answer")
    workflow.add_edge("generate_initial_answer", "search_external")
    workflow.add_edge("search_external", "decide_answer")
    workflow.add_edge("decide_answer", END)
    
    return workflow.compile()