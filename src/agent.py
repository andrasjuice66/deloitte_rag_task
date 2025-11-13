"""LangGraph-based agent for answering Gloomhaven rulebook questions."""

from typing import TypedDict, Annotated, Sequence, Optional
import operator
import json

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_community.llms import HuggingFacePipeline
from langgraph.graph import StateGraph, END
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

from .models import AgentResponse, RuleCategoryEnum
from .rag_system import RAGSystem
from .web_search import WebSearchTool
from .config import Config


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
        model_name: str = None
    ):
        """
        Initialize the Gloomhaven agent.
        
        Args:
            rag_system: Initialized RAG system
            web_search_tool: Web search tool (optional, disabled by default)
            llm: Language model to use (if None, will create from config)
            use_huggingface: Whether to use a local Hugging Face model
            model_name: Name of Hugging Face model (defaults to Config.LLM_MODEL)
        """
        self.rag_system = rag_system
        self.web_search_tool = web_search_tool if Config.ENABLE_WEB_SEARCH else None
        
        # Initialize LLM
        if llm is not None:
            self.llm = llm
        elif use_huggingface or Config.USE_LOCAL_LLM:
            model_name = model_name or Config.LLM_MODEL
            print(f"Loading Hugging Face model: {model_name}")
            self.llm = self._create_huggingface_llm(model_name)
        else:
            # Fallback to OpenAI (not recommended per user request)
            self.llm = ChatOpenAI(
                model=Config.LLM_MODEL,
                temperature=Config.LLM_TEMPERATURE
            )
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _create_huggingface_llm(self, model_name: str):
        """Create a Hugging Face LLM pipeline."""
        # Determine device
        device = 0 if torch.cuda.is_available() else -1
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Using device: {device_str}")
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        
        # Create pipeline
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=Config.LLM_MAX_LENGTH,
            temperature=Config.LLM_TEMPERATURE,
            do_sample=True,
            top_p=0.95,
            device=device
        )
        
        # Wrap in LangChain
        return HuggingFacePipeline(pipeline=pipe)
    
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
        
        messages = prompt.format_messages(context=context, question=question)
        response = self.llm.invoke(messages)
        
        # Parse the response
        try:
            response_text = response.content
            # Try to extract JSON from the response
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_str = response_text.split("```")[1].split("```")[0].strip()
            else:
                json_str = response_text
            
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
        
        messages = prompt.format_messages(
            rulebook_context=rulebook_context,
            web_context=web_context,
            question=question
        )
        response = self.llm.invoke(messages)
        
        # Parse the response
        try:
            response_text = response.content
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_str = response_text.split("```")[1].split("```")[0].strip()
            else:
                json_str = response_text
            
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
        if state.get("needs_web_search", False) and self.web_search_tool is not None:
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
    
    def answer_question(self, question: str) -> AgentResponse:
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
            "needs_web_search": False
        }
        
        final_state = self.graph.invoke(initial_state)
        
        return final_state["answer"]

