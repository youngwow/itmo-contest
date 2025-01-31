from langchain_gigachat.chat_models import GigaChat
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import END, StateGraph
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from typing import TypedDict, List, Optional, Union
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
    tool_calls: Optional[List]

def parse_query(state: AgentState) -> AgentState:
    query = state["query"]
    options = re.findall(r"\n(\d+)\..+", query)
    state["options"] = options if len(options) > 1 else None
    return state

def generate_initial_answer(state: AgentState) -> AgentState:
    llm = GigaChat(
        credentials=os.environ["GIGACHAT_CREDENTIALS"],
        verify_ssl_certs=False
    )
    
    search_tool = TavilySearchResults(
        max_results=3,
        include_answer=True,
        include_raw_content=False,
        include_images=False,
        include_domains=['itmo.ru']
    )
    
    model_with_tools = llm.bind_tools([search_tool])
    
    prompt_template = PromptTemplate.from_template(
        """Ты эксперт по Университету ИТМО. 
        Ответь на вопрос: {query}
        Правила:
        1. Будь максимально точным.
        2. Используй инструмент поиска.
        3. Ответ должен быть кратким и фактологически точным."""
    )
    
    messages = [
        SystemMessage(content="Выполни запрос пользователя, используя доступные инструменты."),
        HumanMessage(content=prompt_template.format(query=state["query"]))
    ]
    
    response = model_with_tools.invoke(messages)
    
    # Обработка ответа
    state["llm_answer"] = response.content
    
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
    answer_template = PromptTemplate.from_template(
    """Сформируй окончательный ответ на основе:
    
    Исходный вопрос: {query}
    Ответ LLM: {llm_answer}
    Варианты: {options_text}
    
    Правила:
    1. Для вопроса с вариантами верни номер ответа (цифрой), соответствующий правильному варианту из списка.
    2. Если ни один вариант не подходит, то верни ответ только null.
    3. Учитывай только соответствие правильного ответа порядку в списке вариантов.
    4. Ответ только цифрой или null."""
    )
    sources = []
    reasoning = []
    
    if state["llm_answer"]:
        reasoning.append(f"GigaChat: {state['llm_answer']}")
    
    if state["search_results"]:
        sources = [HttpUrl(res["url"]) for res in state["search_results"]]
    
    answer = None
    if state["options"]:
        options_text = "\n".join(state["options"])
        llm = GigaChat(
            credentials=os.environ["GIGACHAT_CREDENTIALS"],
            verify_ssl_certs=False,
            temperature=0)
        answer = llm.invoke([
        HumanMessage(content=answer_template.format(
            query=state["query"],
            llm_answer=state["llm_answer"],
            options_text=options_text
        ))]).content
    if answer is not None:
        answer = int(answer) if answer and answer.isdigit() else None
    state["final_answer"] = {
        "id": state["id"],
        "answer": answer,
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