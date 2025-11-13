"""LangGraph-based agent for answering Gloomhaven rulebook questions."""

from typing import TypedDict, Annotated, Sequence, Optional
import operator
import json

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

from .models import AgentResponse, RuleCategoryEnum
from .rag_system import RAGSystem
from .web_search import WebSearchTool
from .config import Config
from .llm import build_llm


def _extract_first_json_object(text: str) -> str:
    """
    Extract the first top-level JSON object from a mixed-content string.
    Handles nested braces and quoted strings.
    Raises ValueError if no balanced JSON object is found.
    """
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object start found")
    depth = 0
    in_string = False
    escape = False
    i = start
    while i < len(text):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    return text[start:end].strip()
        i += 1
    raise ValueError("Unbalanced JSON braces")


class AgentState(TypedDict):
    """State of the agent."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    question: str
    retrieved_docs: Optional[str]
    web_results: Optional[str]
    answer: Optional[AgentResponse]
    needs_web_search: bool


class GloomhavenAgent:
    """Agent for answering questions about Gloomhaven rules using LangGraph."""
    
    def __init__(
        self,
        rag_system: RAGSystem,
        web_search_tool: Optional[WebSearchTool] = None,
        llm: Optional[any] = None,
        use_huggingface: bool = True,
        model_name: Optional[str] = None
    ):
        """
        Initialize the Gloomhaven agent.
        
        Args:
            rag_system: Initialized RAG system
            web_search_tool: Web search tool (optional; if not provided, web search can be skipped)
            llm: Optional pre-initialized HF pipeline (for advanced use)
            use_huggingface: Unused; kept for compatibility (always True)
            model_name: Hugging Face model name (defaults to Config.LLM_MODEL)
        """
        self.rag_system = rag_system
        # Keep provided web_search_tool regardless of config flag; the tool itself
        # will decide whether to use API or fallback.
        self.web_search_tool = web_search_tool
        self.model_name = model_name or Config.LLM_MODEL
        
        # Initialize simple local Hugging Face LLM callable
        if llm is not None:
            self.llm = llm
        else:
            self.llm = build_llm(
                model_name=self.model_name,
                max_new_tokens=max(128, Config.LLM_MAX_LENGTH // 2),
                temperature=Config.LLM_TEMPERATURE,
            )
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _invoke_llm_with_prompt(self, prompt: ChatPromptTemplate, **kwargs) -> str:
        """Format a chat prompt to text and run it via the simple llm(prompt)->str."""
        try:
            prompt_text = prompt.format_prompt(**kwargs).to_string()
        except Exception:
            messages = prompt.format_messages(**kwargs)
            prompt_text = "\n".join([m.content for m in messages])
        
        return self.llm(prompt_text)
    
    def _retrieve_from_rulebook(self, state: AgentState) -> AgentState:
        """Retrieve relevant information from the rulebook."""
        question = state["question"]
        
        # Retrieve relevant documents
        docs = self.rag_system.retrieve(question)
        retrieved_text = "\n\n".join([doc.page_content for doc in docs])
        
        state["retrieved_docs"] = retrieved_text
        state["messages"].append(
            SystemMessage(content=f"Retrieved from rulebook:\n{retrieved_text}")
        )
        
        return state
    
    def _answer_from_rulebook(self, state: AgentState) -> AgentState:
        """Generate an answer based on the rulebook."""
        question = state["question"]
        context = state["retrieved_docs"]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert on the Gloomhaven board game rules. 
        Based on the provided rulebook context, answer the user's question about whether they played a situation correctly.

        Provide a structured response in JSON format with the following fields:
        - explanation: A detailed explanation based on the rules
        - is_correct: Boolean indicating if the user handled the situation correctly
        - category: One of [BoardGameSetup, Combat, Scenario, Character]
        - confidence: A confidence score between 0 and 1
        - source: Always "rulebook" for this response

        If the context doesn't contain enough information to answer confidently (confidence < 0.6), 
        set is_correct to false and indicate in the explanation that more information is needed.

        Context from rulebook:
        {context}"""),
            ("human", "{question}")
        ])
        
        response_text = self._invoke_llm_with_prompt(
            prompt,
            context=context,
            question=question
        )
        
        # Parse the response
        try:
            json_str = _extract_first_json_object(response_text)
            
            response_dict = json.loads(json_str)
            answer = AgentResponse(**response_dict)
            
            state["answer"] = answer
            state["needs_web_search"] = answer.confidence and answer.confidence < 0.6
            
        except Exception as e:
            print(f"Error parsing response: {e}")
            # Create a fallback response
            state["answer"] = AgentResponse(
                explanation="I couldn't find a clear answer in the rulebook.",
                is_correct=False,
                category=RuleCategoryEnum.SCENARIO,
                confidence=0.3,
                source="rulebook"
            )
            state["needs_web_search"] = True
        
        return state
    
    def _search_web(self, state: AgentState) -> AgentState:
        """Search the web for additional information."""
        if self.web_search_tool is None:
            state["web_results"] = "Web search not available."
            return state
        
        question = state["question"]
        search_query = f"Gloomhaven board game rules: {question}"
        
        web_results = self.web_search_tool.search(search_query)
        state["web_results"] = web_results
        
        return state
    
    def _answer_from_web(self, state: AgentState) -> AgentState:
        """Generate an answer incorporating web results."""
        question = state["question"]
        rulebook_context = state["retrieved_docs"]
        web_context = state["web_results"]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert on the Gloomhaven board game rules. 
        Based on the provided rulebook context and web search results, answer the user's question.

        Provide a structured response in JSON format with the following fields:
        - explanation: A detailed explanation based on the rules
        - is_correct: Boolean indicating if the user handled the situation correctly
        - category: One of [BoardGameSetup, Combat, Scenario, Character]
        - confidence: A confidence score between 0 and 1
        - source: "web" since we're incorporating web results

        Rulebook context:
        {rulebook_context}

        Web search results:
        {web_context}"""),
                ("human", "{question}")
            ])
        
        response_text = self._invoke_llm_with_prompt(
            prompt,
            rulebook_context=rulebook_context,
            web_context=web_context,
            question=question
        )
        
        # Parse the response
        try:
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_str = response_text.split("```")[1].split("```")[0].strip()
            else:
                json_str = _extract_first_json_object(response_text)
            
            response_dict = json.loads(json_str)
            answer = AgentResponse(**response_dict)
            answer.source = "web"
            
            state["answer"] = answer
            
        except Exception as e:
            print(f"Error parsing web response: {e}")
            # Keep the previous answer if parsing fails
            if state["answer"]:
                state["answer"].source = "web"
        
        return state
    
    def _should_search_web(self, state: AgentState) -> str:
        """Decide whether to search the web."""
        if state.get("needs_web_search", False):
            return "search_web"
        return "end"
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("retrieve_rulebook", self._retrieve_from_rulebook)
        workflow.add_node("answer_rulebook", self._answer_from_rulebook)
        workflow.add_node("search_web", self._search_web)
        workflow.add_node("answer_web", self._answer_from_web)
        
        # Add edges
        workflow.set_entry_point("retrieve_rulebook")
        workflow.add_edge("retrieve_rulebook", "answer_rulebook")
        
        # Conditional edge: search web if needed
        workflow.add_conditional_edges(
            "answer_rulebook",
            self._should_search_web,
            {
                "search_web": "search_web",
                "end": END
            }
        )
        
        workflow.add_edge("search_web", "answer_web")
        workflow.add_edge("answer_web", END)
        
        return workflow.compile()
    
    def answer_question(self, question: str = None, needs_web_search: bool = False) -> AgentResponse:
        """
        Answer a question about Gloomhaven rules.
        
        Args:
            question: The user's question
            
        Returns:
            Structured answer
        """
        initial_state = {
            "messages": [HumanMessage(content=question)],
            "question": question,
            "retrieved_docs": None,
            "web_results": None,
            "answer": None,
            "needs_web_search": needs_web_search
        }
        
        final_state = self.graph.invoke(initial_state)
        
        return final_state["answer"]

