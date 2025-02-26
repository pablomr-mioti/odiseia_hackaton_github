import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from src.d01_data.data import *
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langchain_core.documents import Document
from typing import List, Tuple
from langgraph.checkpoint.memory import MemorySaver
from agent import AgentWithTools


class ToolRagAgent(AgentWithTools):
    def get_tools(self):
        @tool
        def retrieval(state: MessagesState) -> Tuple[str, List[Document]]:
            """Cuando el usuario hace una pregunta sobre el procedimiento de solicitud de autorizacion, añade un contexto relevante"""
            return self.retrieval(state)
        tools = super().get_tools() + [retrieval]
        
        return tools
               
        
    def setup_workflow(self):
        tool_node = ToolNode(self.get_tools())
        workflow = StateGraph(MessagesState)
        workflow.add_node("agent", self.call_model)
        workflow.add_node("tools", tool_node)
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges("agent", self.should_continue, ["tools", END])
        workflow.add_edge("tools", "agent")
        app = workflow.compile(checkpointer = MemorySaver())
        
        return app