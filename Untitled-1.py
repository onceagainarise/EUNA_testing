# %%
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import Annotated
from typing_extensions import TypedDict

###--------------------tool importing-----------------###
from langchain_community.tools import WikipediaQueryRun, Tool
from langchain_community.utilities import WikipediaAPIWrapper, GoogleSerperAPIWrapper
from langgraph.prebuilt import ToolNode

# %%
class State(TypedDict):
    messages: Annotated[list, add_messages]
    chatbot_answer: str
    tool_answer: str

graph_builder = StateGraph(State)

# %%
import os
from dotenv import load_dotenv
load_dotenv()

# %%
api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(api_key=api_key, model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0.2)

# %%
# Chatbot memory node
def chatbot(state: State):
    answer = llm.invoke(state["messages"])
    return {"messages": [answer], "chatbot_answer": answer.content}

graph_builder.add_node("chatbot", chatbot)

# %%
# Tools
serper_api_key = os.getenv("SERP_API_KEY")
google = GoogleSerperAPIWrapper(serper_api_key=serper_api_key)
google_tool = Tool(
    name="Google Search",
    func=google.run,
    description="Use this tool when the user asks about current events, live updates, or real-time data."
)

wiki_api_wrapper = WikipediaAPIWrapper()
wikipedia_tool = WikipediaQueryRun(api_wrapper=wiki_api_wrapper)

tool_node = ToolNode(tools=[google_tool, wikipedia_tool])
graph_builder.add_node("tools", tool_node)

# %%
# Router node (LLM decides: memory / tools / both)
def router(state: State):
    query = state["messages"][-1].content

    router_prompt = f"""
You are a smart router. Decide how to answer this query:

"{query}"

Options:
- "memory" → If the question is static, historical, or definitional (encyclopedia-like).
- "tools" → If the question needs current, real-time, or frequently changing data.
- "both" → If both memory and external tools are useful.

Answer with one word only: memory, tools, or both.
    """

    decision = llm.invoke(router_prompt).content.strip().lower()
    if decision not in ["memory", "tools", "both"]:
        decision = {"messages":"memory"}  # fallback
    return decision

graph_builder.add_node("router", router)

# %%
# Comparison node (merge chatbot + tools answers)
def compare_answers(state: State):
    chatbot_ans = state.get("chatbot_answer", "")
    tool_ans = state.get("tool_answer", "")

    prompt = f"""
You are an assistant. You have two possible answers to the user's question:

Answer from memory (may be outdated):
{chatbot_ans}

Answer from external tools (likely more up-to-date):
{tool_ans}

Please give the most relevant, correct, and up-to-date answer to the user.
If the tool answer is empty or irrelevant, fallback to memory.
"""
    final = llm.invoke(prompt)
    return {"messages": [final]}

graph_builder.add_node("compare", compare_answers)

# %%
# Graph Edges
graph_builder.add_edge(START, "router")

# Map router decision to nodes
def route_decision(state: State):
    decision = router(state)
    if decision == "memory":
        return "chatbot"
    elif decision == "tools":
        return "tools"
    elif decision == "both":
        return "compare"
    else:
        return "chatbot"  # fallback

graph_builder.add_conditional_edges(
    "router",
    route_decision
)

# If "both", run both chatbot + tools, then merge
graph_builder.add_edge("chatbot", "compare")
graph_builder.add_edge("tools", "compare")

# End connections
graph_builder.add_edge("chatbot", END)
graph_builder.add_edge("tools", END)
graph_builder.add_edge("compare", END)

graph = graph_builder.compile()

# %%
from IPython.display import Image, display
try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except:
    pass

# %%
def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)

while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        stream_graph_updates(user_input)
    except:
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break
