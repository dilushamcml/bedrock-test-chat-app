"""
LangGraph ReAct Agent Implementation with Tool Calling
Advanced chat agent with reasoning, action, and observation capabilities
"""

import logging
import asyncio
import json
import math
import requests
from typing import Dict, List, TypedDict, Annotated, Optional, Any, AsyncGenerator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from bedrock_client import BedrockClient, BedrockError
from config import Config

# Configure logging
logger = logging.getLogger(__name__)

class ReActState(TypedDict):
    """Enhanced ReAct agent state schema"""
    messages: Annotated[List[BaseMessage], add_messages]
    chat_type: str
    model_id: str
    
    # ReAct specific fields
    current_thought: str
    next_action: str
    action_input: str
    observation: str
    final_answer: str
    
    # Iteration tracking
    iteration_count: int
    max_iterations: int
    
    # Tool tracking
    tools_used: List[str]
    tool_results: List[Dict]
    
    # Context management
    total_messages: int
    context_messages: int
    estimated_tokens: int
    cache_used: bool

class ChatAgentError(Exception):
    """Custom exception for chat agent operations"""
    pass

class ToolExecutor:
    """Tool execution engine for ReAct agent"""
    
    def __init__(self):
        self.tools = {
            "web_search": self._web_search,
            "calculator": self._calculator,
            "code_executor": self._code_executor,
            "weather_api": self._weather_api,
            "file_reader": self._file_reader,
            "database_query": self._database_query
        }
    
    async def execute_tool(self, tool_name: str, tool_input: str) -> str:
        """Execute a tool and return the result"""
        if tool_name not in self.tools:
            return f"Error: Unknown tool '{tool_name}'. Available tools: {list(self.tools.keys())}"
        
        try:
            result = await self.tools[tool_name](tool_input)
            return result
        except Exception as e:
            logger.error(f"Tool execution failed for {tool_name}: {str(e)}")
            return f"Error executing {tool_name}: {str(e)}"
    
    async def _web_search(self, query: str) -> str:
        """Web search tool - mock implementation"""
        # In production, integrate with real search API (Google, Bing, etc.)
        try:
            # Mock web search results
            mock_results = [
                f"Search result 1 for '{query}': Relevant information about {query}",
                f"Search result 2 for '{query}': Additional details and context",
                f"Search result 3 for '{query}': Related topics and insights"
            ]
            return f"Web search results for '{query}':\n" + "\n".join(mock_results)
        except Exception as e:
            return f"Web search failed: {str(e)}"
    
    async def _calculator(self, expression: str) -> str:
        """Calculator tool for mathematical operations"""
        try:
            # Safe evaluation of mathematical expressions
            allowed_names = {
                k: v for k, v in math.__dict__.items() if not k.startswith("__")
            }
            allowed_names.update({"abs": abs, "round": round, "min": min, "max": max})
            
            # Remove any potentially dangerous operations
            expression = expression.replace("__", "").replace("import", "").replace("exec", "")
            
            result = eval(expression, {"__builtins__": {}}, allowed_names)
            return f"Calculation result: {expression} = {result}"
        except Exception as e:
            return f"Calculation failed: {str(e)}"
    
    async def _code_executor(self, code: str) -> str:
        """Code execution tool - sandboxed Python execution"""
        try:
            # In production, use a proper sandboxed environment
            # This is a simplified version for demonstration
            
            # Create a restricted execution environment
            restricted_globals = {
                "__builtins__": {
                    "print": print,
                    "len": len,
                    "str": str,
                    "int": int,
                    "float": float,
                    "list": list,
                    "dict": dict,
                    "range": range,
                    "sum": sum,
                    "max": max,
                    "min": min,
                    "abs": abs,
                    "round": round
                }
            }
            
            # Capture output
            import io
            import sys
            old_stdout = sys.stdout
            sys.stdout = captured_output = io.StringIO()
            
            try:
                exec(code, restricted_globals)
                output = captured_output.getvalue()
                return f"Code executed successfully:\n{output}" if output else "Code executed successfully (no output)"
            finally:
                sys.stdout = old_stdout
                
        except Exception as e:
            return f"Code execution failed: {str(e)}"
    
    async def _weather_api(self, location: str) -> str:
        """Weather API tool - mock implementation"""
        try:
            # Mock weather data
            import random
            temp = random.randint(15, 35)
            conditions = random.choice(["sunny", "cloudy", "rainy", "partly cloudy"])
            
            return f"Weather in {location}: {temp}°C, {conditions}"
        except Exception as e:
            return f"Weather lookup failed: {str(e)}"
    
    async def _file_reader(self, file_path: str) -> str:
        """File reader tool - read local files safely"""
        try:
            # Security check - only allow certain file types and paths
            allowed_extensions = ['.txt', '.md', '.json', '.csv', '.log']
            if not any(file_path.endswith(ext) for ext in allowed_extensions):
                return f"Error: File type not allowed. Allowed types: {allowed_extensions}"
            
            # Read file content (with size limit)
            max_size = 10000  # 10KB limit
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read(max_size)
                if len(content) == max_size:
                    content += "\n... (file truncated)"
                
            return f"File content of {file_path}:\n{content}"
        except Exception as e:
            return f"File reading failed: {str(e)}"
    
    async def _database_query(self, query: str) -> str:
        """Database query tool - mock implementation"""
        try:
            # Mock database query results
            return f"Database query result for: {query}\nMock result: 42 rows affected"
        except Exception as e:
            return f"Database query failed: {str(e)}"

class ChatAgent:
    """Advanced ReAct-based chat agent with tool calling capabilities"""
    
    def __init__(self, bedrock_client: BedrockClient):
        self.bedrock_client = bedrock_client
        self.memory = MemorySaver()
        self.tool_executor = ToolExecutor()
        self.max_tokens = Config.MAX_CONTEXT_TOKENS
        self.max_context_messages = Config.MAX_CONTEXT_MESSAGES
        
        # Enhanced system prompts with tool awareness
        self.system_prompts = {
            "General Chat": """You are a helpful AI assistant. Provide clear, accurate, and helpful responses. 
You have access to various tools when needed. Think step by step and use tools when they would be helpful.""",
            
            "Code Assistant": """You are an expert programming assistant. Help with coding questions, debugging, and best practices.
You can execute code using the code_executor tool to test and demonstrate solutions.""",
            
            "Creative Writing": """You are a creative writing assistant. Help with storytelling, character development, and creative expression.
You can search for inspiration or reference materials using the web_search tool when needed.""",
            
            "Analysis & Research": """You are a research and analysis assistant. Provide thorough, well-structured analyses.
Use web_search for current information, calculator for computations, and other tools as needed.""",
            
            "Question & Answer": """You are a knowledgeable Q&A assistant. Provide direct, comprehensive answers.
Use appropriate tools to gather information, perform calculations, or verify facts.""",
            
            "Tool-Assisted Research": """You are an advanced research assistant with access to multiple tools.
Think step by step, use tools strategically, and provide comprehensive, well-researched responses."""
        }
        
        # Build the ReAct workflow
        self.graph = self._build_react_graph()
        logger.info("ReAct ChatAgent initialized with tool calling support")
    
    def _get_system_prompt(self, chat_type: str) -> str:
        """Get system prompt for the given chat type"""
        return self.system_prompts.get(chat_type, self.system_prompts["General Chat"])
    
    def _build_react_graph(self) -> StateGraph:
        """Build ReAct workflow with tool calling"""
        try:
            workflow = StateGraph(ReActState)
            
            # Add ReAct nodes
            workflow.add_node("reasoner", self._reasoning_node)
            workflow.add_node("action_planner", self._action_planning_node)
            workflow.add_node("tool_executor", self._tool_execution_node)
            workflow.add_node("observer", self._observation_node)
            workflow.add_node("responder", self._response_node)
            
            # Set entry point
            workflow.set_entry_point("reasoner")
            
            # Add conditional edges
            workflow.add_conditional_edges(
                "reasoner",
                self._should_use_tool,
                {
                    "use_tool": "action_planner",
                    "respond": "responder"
                }
            )
            
            workflow.add_edge("action_planner", "tool_executor")
            workflow.add_edge("tool_executor", "observer")
            
            workflow.add_conditional_edges(
                "observer",
                self._should_continue_reasoning,
                {
                    "continue": "reasoner",
                    "finish": "responder"
                }
            )
            
            workflow.set_finish_point("responder")
            
            compiled_graph = workflow.compile(checkpointer=self.memory)
            logger.info("ReAct workflow compiled successfully")
            return compiled_graph
            
        except Exception as e:
            logger.error(f"Failed to build ReAct workflow: {str(e)}")
            raise ChatAgentError(f"Graph building failed: {str(e)}")
    
    async def _reasoning_node(self, state: ReActState) -> Dict:
        """ReAct Reasoning Node - Think about what to do"""
        try:
            messages = state.get("messages", [])
            chat_type = state.get("chat_type", "General Chat")
            model_id = state.get("model_id")
            if not model_id:
                # Get first available model from bedrock client
                available_models = self.bedrock_client.get_available_models()
                if available_models:
                    model_id = list(available_models.values())[0]
                else:
                    raise ChatAgentError("No available models found")
            
            iteration_count = state.get("iteration_count", 0)
            
            # Get the latest user message or continue reasoning
            if iteration_count == 0:
                # First iteration - analyze user request
                last_message = messages[-1].content if messages else ""
                context = f"User request: {last_message}"
            else:
                # Subsequent iterations - continue reasoning with observations
                observation = state.get("observation", "")
                context = f"Previous observation: {observation}\nContinue reasoning..."
            
            reasoning_prompt = f"""
            {self._get_system_prompt(chat_type)}
            
            Current situation: {context}
            
            Available tools:
            - web_search(query): Search the internet for current information
            - calculator(expression): Perform mathematical calculations  
            - code_executor(code): Execute Python code safely
            - weather_api(location): Get weather information
            - file_reader(path): Read local files
            - database_query(sql): Query databases
            
            Think step by step:
            1. What is the user asking for?
            2. What information do I need to provide a complete answer?
            3. Do I need to use any tools to get this information?
            4. If so, which tool would be most appropriate?
            
            Provide your reasoning and decide whether you need to use a tool or can respond directly.
            """
            
            # Use LangChain ChatBedrock for reasoning
            chat_model = self.bedrock_client.get_chat_model(
                model_id=model_id,
                temperature=0.7,
                max_tokens=1000
            )
            
            langchain_messages = [HumanMessage(content=reasoning_prompt)]
            response = chat_model.invoke(langchain_messages)
            thought = response.content if response.content else "Unable to generate reasoning"
            
            return {
                "current_thought": thought,
                "iteration_count": iteration_count + 1,
                "max_iterations": Config.MAX_TOOL_ITERATIONS,
                "model_id": model_id  # Ensure model_id is passed through
            }
            
        except Exception as e:
            logger.error(f"Error in reasoning node: {str(e)}")
            return {
                "current_thought": f"Error in reasoning: {str(e)}",
                "iteration_count": state.get("iteration_count", 0) + 1
            }
    
    async def _stream_reasoning_node(self, state: ReActState) -> AsyncGenerator[str, None]:
        """ReAct Reasoning Node with REAL-TIME streaming - Think about what to do"""
        try:
            messages = state.get("messages", [])
            chat_type = state.get("chat_type", "General Chat")
            model_id = state.get("model_id")
            if not model_id:
                # Get first available model from bedrock client
                available_models = self.bedrock_client.get_available_models()
                if available_models:
                    model_id = list(available_models.values())[0]
                else:
                    raise ChatAgentError("No available models found")
            
            iteration_count = state.get("iteration_count", 0)
            
            # Get the latest user message or continue reasoning
            if iteration_count == 0:
                # First iteration - analyze user request
                last_message = messages[-1].content if messages else ""
                context = f"User request: {last_message}"
            else:
                # Subsequent iterations - continue reasoning with observations
                observation = state.get("observation", "")
                context = f"Previous observation: {observation}\nContinue reasoning..."
            
            reasoning_prompt = f"""
            {self._get_system_prompt(chat_type)}
            
            Current situation: {context}
            
            Available tools:
            - web_search(query): Search the internet for current information
            - calculator(expression): Perform mathematical calculations  
            - code_executor(code): Execute Python code safely
            - weather_api(location): Get weather information
            - file_reader(path): Read local files
            - database_query(sql): Query databases
            
            Think step by step:
            1. What is the user asking for?
            2. What information do I need to provide a complete answer?
            3. Do I need to use any tools to get this information?
            4. If so, which tool would be most appropriate?
            
            Provide your reasoning and decide whether you need to use a tool or can respond directly.
            """
            
            # Use LangChain ChatBedrock for STREAMING reasoning
            chat_model = self.bedrock_client.get_chat_model(
                model_id=model_id,
                temperature=0.7,
                max_tokens=1000
            )
            
            langchain_messages = [HumanMessage(content=reasoning_prompt)]
            
            # Stream the reasoning process character by character
            full_thought = ""
            for chunk in chat_model.stream(langchain_messages):
                if chunk.content:
                    full_thought += chunk.content
                    yield chunk.content  # Yield each character/token as it comes
            
            # Update state with complete thought
            state.update({
                "current_thought": full_thought,
                "iteration_count": iteration_count + 1,
                "max_iterations": Config.MAX_TOOL_ITERATIONS,
                "model_id": model_id
            })
            
        except Exception as e:
            logger.error(f"Error in streaming reasoning node: {str(e)}")
            error_thought = f"Error in reasoning: {str(e)}"
            # Stream the error message
            for char in error_thought:
                yield char
                await asyncio.sleep(0.05)  # Simulate typing for errors
            
            state.update({
                "current_thought": error_thought,
                "iteration_count": state.get("iteration_count", 0) + 1
            })

    async def _action_planning_node(self, state: ReActState) -> Dict:
        """ReAct Action Planning Node - Decide which tool to use"""
        try:
            thought = state.get("current_thought", "")
            model_id = state.get("model_id")
            if not model_id:
                # Get first available model from bedrock client
                available_models = self.bedrock_client.get_available_models()
                if available_models:
                    model_id = list(available_models.values())[0]
                else:
                    raise ChatAgentError("No available models found")
            
            action_prompt = f"""
            Based on this reasoning: {thought}
            
            Which tool should I use and with what input?
            
            Available tools:
            - web_search(query): Search the internet
            - calculator(expression): Calculate math expressions
            - code_executor(code): Execute Python code
            - weather_api(location): Get weather info
            - file_reader(path): Read files
            - database_query(sql): Query database
            
            Respond in this exact format:
            TOOL: tool_name
            INPUT: specific_input_for_tool
            
            Or if no tool is needed:
            TOOL: none
            INPUT: ready_to_respond
            """
            
            # Use LangChain ChatBedrock for action planning
            chat_model = self.bedrock_client.get_chat_model(
                model_id=model_id,
                temperature=0.7,
                max_tokens=500
            )
            
            langchain_messages = [HumanMessage(content=action_prompt)]
            response = chat_model.invoke(langchain_messages)
            action_decision = response.content if response.content else "TOOL: none\nINPUT: ready_to_respond"
            
            # Parse the action decision
            lines = action_decision.strip().split('\n')
            tool_name = "none"
            tool_input = "ready_to_respond"
            
            for line in lines:
                if line.startswith("TOOL:"):
                    tool_name = line.split(":", 1)[1].strip()
                elif line.startswith("INPUT:"):
                    tool_input = line.split(":", 1)[1].strip()
            
            return {
                "next_action": tool_name,
                "action_input": tool_input,
                "model_id": model_id  # Ensure model_id is passed through
            }
            
        except Exception as e:
            logger.error(f"Error in action planning: {str(e)}")
            return {
                "next_action": "none",
                "action_input": "ready_to_respond"
            }
    
    async def _stream_action_planning_node(self, state: ReActState) -> AsyncGenerator[str, None]:
        """ReAct Action Planning Node with REAL-TIME streaming - Decide which tool to use"""
        try:
            thought = state.get("current_thought", "")
            model_id = state.get("model_id")
            if not model_id:
                # Get first available model from bedrock client
                available_models = self.bedrock_client.get_available_models()
                if available_models:
                    model_id = list(available_models.values())[0]
                else:
                    raise ChatAgentError("No available models found")
            
            action_prompt = f"""
            Based on this reasoning: {thought}
            
            Which tool should I use and with what input?
            
            Available tools:
            - web_search(query): Search the internet
            - calculator(expression): Calculate math expressions
            - code_executor(code): Execute Python code
            - weather_api(location): Get weather info
            - file_reader(path): Read files
            - database_query(sql): Query database
            
            Respond in this exact format:
            TOOL: tool_name
            INPUT: specific_input_for_tool
            
            Or if no tool is needed:
            TOOL: none
            INPUT: ready_to_respond
            """
            
            # Use LangChain ChatBedrock for STREAMING action planning
            chat_model = self.bedrock_client.get_chat_model(
                model_id=model_id,
                temperature=0.7,
                max_tokens=500
            )
            
            langchain_messages = [HumanMessage(content=action_prompt)]
            
            # Stream the action planning process
            full_response = ""
            for chunk in chat_model.stream(langchain_messages):
                if chunk.content:
                    full_response += chunk.content
                    yield chunk.content  # Yield each token as it comes
            
            # Parse the streamed response
            lines = full_response.strip().split('\n')
            next_action = "none"
            action_input = ""
            
            for line in lines:
                if line.startswith("TOOL:"):
                    next_action = line.replace("TOOL:", "").strip().lower()
                elif line.startswith("INPUT:"):
                    action_input = line.replace("INPUT:", "").strip()
            
            # Update state with parsed action
            state.update({
                "next_action": next_action,
                "action_input": action_input,
                "model_id": model_id
            })
            
        except Exception as e:
            logger.error(f"Error in streaming action planning: {str(e)}")
            error_msg = f"Error in action planning: {str(e)}"
            # Stream the error message
            for char in error_msg:
                yield char
                await asyncio.sleep(0.05)
            
            state.update({
                "next_action": "none",
                "action_input": f"Error: {str(e)}"
            })

    async def _tool_execution_node(self, state: ReActState) -> Dict:
        """ReAct Tool Execution Node - Execute the chosen tool"""
        try:
            action = state.get("next_action", "none")
            action_input = state.get("action_input", "")
            
            if action == "none":
                observation = "No tool execution needed"
            else:
                # Execute the tool
                observation = await self.tool_executor.execute_tool(action, action_input)
            
            # Track tool usage
            tools_used = state.get("tools_used", [])
            tool_results = state.get("tool_results", [])
            
            tools_used.append(action)
            tool_results.append({
                "tool": action,
                "input": action_input,
                "output": observation
            })
            
            return {
                "observation": observation,
                "tools_used": tools_used,
                "tool_results": tool_results
            }
            
        except Exception as e:
            logger.error(f"Error in tool execution: {str(e)}")
            return {
                "observation": f"Tool execution failed: {str(e)}",
                "tools_used": state.get("tools_used", []),
                "tool_results": state.get("tool_results", [])
            }
    
    async def _observation_node(self, state: ReActState) -> Dict:
        """ReAct Observation Node - Analyze tool results"""
        try:
            tool_results = state.get("tool_results", [])
            model_id = state.get("model_id")
            if not model_id:
                # Get first available model from bedrock client
                available_models = self.bedrock_client.get_available_models()
                if available_models:
                    model_id = list(available_models.values())[0]
                else:
                    raise ChatAgentError("No available models found")
            
            if not tool_results:
                return {"observation": "No tool results to analyze"}
            
            # Get the latest tool result
            latest_result = tool_results[-1]
            
            observation_prompt = f"""
            Analyze this tool result and determine what it means for answering the user's question:
            
            Tool used: {latest_result['tool']}
            Tool input: {latest_result['input']}
            Tool output: {latest_result['output']}
            
            What does this result tell us? Is it sufficient to answer the user's question, or do we need more information?
            Provide a brief analysis of what we learned.
            """
            
            # Use LangChain ChatBedrock for observation
            chat_model = self.bedrock_client.get_chat_model(
                model_id=model_id,
                temperature=0.7,
                max_tokens=500
            )
            
            langchain_messages = [HumanMessage(content=observation_prompt)]
            response = chat_model.invoke(langchain_messages)
            observation = response.content if response.content else "Unable to analyze tool results"
            
            return {
                "observation": observation,
                "model_id": model_id  # Ensure model_id is passed through
            }
            
        except Exception as e:
            logger.error(f"Error in observation: {str(e)}")
            return {
                "observation": f"Error analyzing tool results: {str(e)}"
            }
    
    async def _response_node(self, state: ReActState) -> Dict:
        """ReAct Response Node - Generate final response"""
        try:
            messages = state.get("messages", [])
            chat_type = state.get("chat_type", "General Chat")
            model_id = state.get("model_id")
            if not model_id:
                # Get first available model from bedrock client
                available_models = self.bedrock_client.get_available_models()
                if available_models:
                    model_id = list(available_models.values())[0]
                else:
                    raise ChatAgentError("No available models found")
            
            current_thought = state.get("current_thought", "")
            tool_results = state.get("tool_results", [])
            
            # Get original user message
            user_message = messages[-1].content if messages else ""
            
            # Prepare context with tool results
            tool_context = ""
            if tool_results:
                tool_context = "\n\nTool Results:\n"
                for result in tool_results:
                    tool_context += f"- {result['tool']}: {result['output']}\n"
            
            response_prompt = f"""
            {self._get_system_prompt(chat_type)}
            
            User question: {user_message}
            
            Your reasoning process: {current_thought}
            {tool_context}
            
            Based on your reasoning and any tool results, provide a comprehensive, helpful response to the user.
            Be clear, accurate, and reference any tool results that informed your answer.
            """
            
            # Use LangChain ChatBedrockConverse for final response
            chat_model = self.bedrock_client.get_chat_model(
                model_id=model_id,
                temperature=0.7,
                max_tokens=2000
            )
            
            langchain_messages = [HumanMessage(content=response_prompt)]
            response = chat_model.invoke(langchain_messages)
            final_response = response.content if response.content else "Unable to generate response"
            
            # Create AI message
            ai_message = AIMessage(content=final_response)
            
            # Update state with final response
            updated_messages = messages + [ai_message]
            
            return {
                "messages": updated_messages,
                "final_answer": final_response,
                "model_id": model_id  # Ensure model_id is passed through
            }
            
        except Exception as e:
            logger.error(f"Error in response generation: {str(e)}")
            error_message = AIMessage(content=f"I apologize, but I encountered an error generating a response: {str(e)}")
            return {
                "messages": state.get("messages", []) + [error_message],
                "final_answer": f"Error: {str(e)}"
            }
    
    async def _stream_response_node(self, state: ReActState) -> AsyncGenerator[str, None]:
        """ReAct Response Node with REAL-TIME streaming - Generate final response"""
        try:
            messages = state.get("messages", [])
            chat_type = state.get("chat_type", "General Chat")
            model_id = state.get("model_id")
            if not model_id:
                # Get first available model from bedrock client
                available_models = self.bedrock_client.get_available_models()
                if available_models:
                    model_id = list(available_models.values())[0]
                else:
                    raise ChatAgentError("No available models found")
            
            current_thought = state.get("current_thought", "")
            tool_results = state.get("tool_results", [])
            
            # Get original user message
            user_message = messages[-1].content if messages else ""
            
            # Prepare context with tool results
            tool_context = ""
            if tool_results:
                tool_context = "\n\nTool Results:\n"
                for result in tool_results:
                    tool_context += f"- {result['tool']}: {result['output']}\n"
            
            response_prompt = f"""
            {self._get_system_prompt(chat_type)}
            
            User question: {user_message}
            
            Your reasoning process: {current_thought}
            {tool_context}
            
            Based on your reasoning and any tool results, provide a comprehensive, helpful response to the user.
            Be clear, accurate, and reference any tool results that informed your answer.
            """
            
            # Use LangChain ChatBedrock for STREAMING final response
            chat_model = self.bedrock_client.get_chat_model(
                model_id=model_id,
                temperature=0.7,
                max_tokens=2000
            )
            
            langchain_messages = [HumanMessage(content=response_prompt)]
            
            # Stream the final response generation
            final_response = ""
            for chunk in chat_model.stream(langchain_messages):
                if chunk.content:
                    final_response += chunk.content
                    yield chunk.content  # Yield each token as it comes
            
            # Create AI message
            ai_message = AIMessage(content=final_response)
            
            # Update state with final response
            updated_messages = messages + [ai_message]
            
            state.update({
                "messages": updated_messages,
                "final_answer": final_response,
                "model_id": model_id
            })
            
        except Exception as e:
            logger.error(f"Error in streaming response generation: {str(e)}")
            error_message = f"I apologize, but I encountered an error generating a response: {str(e)}"
            # Stream the error message
            for char in error_message:
                yield char
                await asyncio.sleep(0.05)
            
            error_ai_message = AIMessage(content=error_message)
            state.update({
                "messages": state.get("messages", []) + [error_ai_message],
                "final_answer": error_message
            })
    
    def _should_use_tool(self, state: ReActState) -> str:
        """Decision function: should we use a tool or respond directly?"""
        thought = state.get("current_thought", "").lower()
        iteration_count = state.get("iteration_count", 0)
        
        # Check if we've reached max iterations
        if iteration_count >= Config.MAX_TOOL_ITERATIONS:
            return "respond"
        
        # Check if reasoning indicates tool use
        tool_indicators = [
            "search", "calculate", "code", "weather", "file", "database",
            "need to", "should use", "tool", "execute", "run", "query"
        ]
        
        if any(indicator in thought for indicator in tool_indicators):
            return "use_tool"
        
        return "respond"
    
    def _should_continue_reasoning(self, state: ReActState) -> str:
        """Decision function: should we continue reasoning or finish?"""
        iteration_count = state.get("iteration_count", 0)
        max_iterations = state.get("max_iterations", Config.MAX_TOOL_ITERATIONS)
        observation = state.get("observation", "").lower()
        
        # Stop if we've reached max iterations
        if iteration_count >= max_iterations:
            return "finish"
        
        # Stop if observation indicates completion
        completion_indicators = [
            "complete", "finished", "done", "enough information", 
            "ready to respond", "no additional", "sufficient"
        ]
        
        if any(indicator in observation for indicator in completion_indicators):
            return "finish"
        
        # Continue if observation suggests more work needed
        continue_indicators = [
            "need more", "additional", "also need", "should also", 
            "next step", "furthermore", "however"
        ]
        
        if any(indicator in observation for indicator in continue_indicators):
            return "continue"
        
        return "finish"
    
    def _estimate_token_count(self, messages: List[BaseMessage]) -> int:
        """Estimate token count for messages"""
        total_tokens = 0
        for message in messages:
            if hasattr(message, 'content'):
                content = str(message.content)
                char_tokens = len(content) / 4
                word_tokens = len(content.split()) * 1.3
                estimated = max(char_tokens, word_tokens)
                total_tokens += int(estimated)
        return total_tokens
    
    def _get_system_prompt(self, chat_type: str) -> str:
        """Get system prompt for chat type"""
        return self.system_prompts.get(chat_type, self.system_prompts["General Chat"])
    
    def get_context_info(self, state: Dict) -> Dict:
        """Get context information for debugging"""
        return {
            "total_messages": state.get("total_messages", 0),
            "context_messages": state.get("context_messages", 0),
            "estimated_tokens": state.get("estimated_tokens", 0),
            "cache_used": state.get("cache_used", False),
            "chat_type": state.get("chat_type", "Unknown"),
            "model_id": state.get("model_id", "Unknown"),
            "iteration_count": state.get("iteration_count", 0),
            "tools_used": state.get("tools_used", []),
            "tool_results": state.get("tool_results", []),
            "current_thought": state.get("current_thought", ""),
            "next_action": state.get("next_action", ""),
            "action_input": state.get("action_input", ""),
            "observation": state.get("observation", ""),
            "final_answer": state.get("final_answer", "")
        }

    def _agent_node(self, state: ReActState, config: RunnableConfig) -> ReActState:
        """
        Main agent node that processes messages and generates responses
        Uses LangChain ChatBedrockConverse for model interaction
        """
        try:
            # Get the current model and messages
            model_id = state.get("model_id")
            if not model_id:
                # Get first available model from bedrock client
                available_models = self.bedrock_client.get_available_models()
                if available_models:
                    model_id = list(available_models.values())[0]
                else:
                    raise ChatAgentError("No available models found")
            
            messages = state.get("messages", [])
            
            # Trim messages to stay within context limits
            trimmed_messages = self._trim_messages(messages)
            
            # Create the prompt with system message
            system_prompt = self._create_system_prompt(state.get("chat_type", "general"))
            
            # Convert to LangChain messages
            langchain_messages = [SystemMessage(content=system_prompt)]
            langchain_messages.extend(trimmed_messages)
            
            # Use the bedrock client to get a chat model and invoke it
            chat_model = self.bedrock_client.get_chat_model(
                model_id=model_id,
                temperature=0.7,
                max_tokens=4000
            )
            
            # Generate response
            response = chat_model.invoke(langchain_messages)
            
            # Create AI message from response
            ai_message = AIMessage(content=response.content if response.content else "I apologize, but I couldn't generate a response.")
            
            # Update state
            updated_messages = messages + [ai_message]
            
            return {
                **state,
                "messages": updated_messages,
                "total_messages": len(updated_messages),
                "context_messages": len(trimmed_messages),
                "estimated_tokens": self._estimate_tokens(updated_messages),
                "cache_used": len(trimmed_messages) < len(messages)
            }
            
        except Exception as e:
            logger.error(f"Error in agent node: {str(e)}")
            # Return error state
            error_message = AIMessage(content=f"I apologize, but I encountered an error: {str(e)}")
            return {
                **state,
                "messages": state.get("messages", []) + [error_message],
                "total_messages": len(state.get("messages", [])) + 1,
                "context_messages": 0,
                "estimated_tokens": 0,
                "cache_used": False
            }