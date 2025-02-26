import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from src.d01_data.data import *
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langchain_core.documents import Document
from typing import List, Tuple
from langgraph.checkpoint.memory import MemorySaver
from agent import AgentWithTools


class RagAgent(AgentWithTools):
    def retrieval(self, state: MessagesState) -> Tuple[str, List[Document]]:
        retrieval_result = super().retrieval(state)
        if isinstance(retrieval_result, str):
            messages = state["messages"][:-1] + [HumanMessage(content=retrieval_result)] + [RemoveMessage(id = state["messages"][-1].id)]
            return {"messages": messages}
        else:
            return state
          
        
    def setup_workflow(self):
        tool_node = ToolNode(self.get_tools())
        workflow = StateGraph(MessagesState)
        workflow.add_node("agent", self.call_model)
        # workflow.add_node("retrieval", self.retrieval)
        workflow.add_node("tools", tool_node)
        workflow.add_edge(START, "agent")
        # workflow.add_edge("retrieval", "agent")
        workflow.add_conditional_edges("agent", self.should_continue, ["tools", END])
        workflow.add_edge("tools", "agent")
        app = workflow.compile(checkpointer = MemorySaver())
        
        return app