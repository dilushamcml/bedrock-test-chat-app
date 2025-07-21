"""
Flask Web Application for Bedrock Chat Assistant
Complete chat interface with LangGraph integration and AWS Bedrock support
Enhanced with streaming responses and modern UI
"""

from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_socketio import SocketIO, emit
import asyncio
import logging
import json
import uuid
import time
from datetime import datetime
from typing import Dict, List, Optional
from langchain_core.messages import HumanMessage, AIMessage

# Import our custom modules
from config import Config
from database import ChatDatabase, DatabaseError
from bedrock_client import BedrockClient, BedrockError
from chat_agent import ChatAgent, ChatAgentError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Flask Application Setup
# ============================================================================

app = Flask(__name__)
app.secret_key = Config.get_environment_info().get('version', 'bedrock-chat-secret-key')
socketio = SocketIO(app, cors_allowed_origins="*", logger=True, engineio_logger=True)

# Global instances
database = None
bedrock_client = None
chat_agent = None
available_models = {}

def initialize_app():
    """Initialize the Flask application with database and clients"""
    global database, bedrock_client, chat_agent, available_models
    
    try:
        # Initialize database
        database = ChatDatabase()
        logger.info("Database initialized successfully")
        
        # Initialize Bedrock client
        bedrock_client = BedrockClient()
        logger.info("Bedrock client initialized")
        
        # Validate credentials and get models
        is_valid, message = bedrock_client.validate_credentials()
        if is_valid:
            available_models = bedrock_client.get_available_models()
            chat_agent = ChatAgent(bedrock_client)
            logger.info("Chat agent initialized successfully")
        else:
            logger.warning(f"AWS credentials not valid: {message}")
            
    except Exception as e:
        logger.error(f"Failed to initialize application: {str(e)}")
        raise

def get_preferred_model():
    """Get the preferred model from available models"""
    if not available_models:
        return None, None
    
    # Try to find a preferred model
    for preferred in Config.PREFERRED_MODELS:
        for display_name, model_id in available_models.items():
            if model_id == preferred:
                return display_name, model_id
    
    # Return first available model
    first_display_name = list(available_models.keys())[0]
    return first_display_name, available_models[first_display_name]

# Initialize the application
try:
    initialize_app()
    logger.info("Flask application initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Flask application: {str(e)}")

# ============================================================================
# Routes
# ============================================================================

@app.route('/')
def welcome():
    """Welcome page with system status"""
    try:
        # Get system status
        credentials_valid = False
        available_models_dict = {}
        chat_agent_ready = False
        
        if bedrock_client:
            credentials_valid = bedrock_client.is_credentials_valid()
            if credentials_valid:
                available_models_dict = bedrock_client.get_available_models()
        
        if chat_agent:
            chat_agent_ready = True
        
        # Get library versions
        try:
            import langchain
            import langgraph
            langchain_version = langchain.__version__
            langgraph_version = langgraph.__version__
        except:
            langchain_version = "Unknown"
            langgraph_version = "Unknown"
        
        return render_template('welcome.html',
                             credentials_valid=credentials_valid,
                             available_models=available_models_dict,
                             chat_agent=chat_agent_ready,
                             langchain_version=langchain_version,
                             langgraph_version=langgraph_version)
    except Exception as e:
        logger.error(f"Error in welcome route: {str(e)}")
        return render_template('welcome.html',
                             credentials_valid=False,
                             available_models={},
                             chat_agent=False,
                             langchain_version="Unknown",
                             langgraph_version="Unknown")

@app.route('/chat')
def chat():
    """Main chat interface"""
    try:
        # Initialize session
        session_id = session.get('session_id')
        if not session_id:
            session_id = str(uuid.uuid4())
            session['session_id'] = session_id
        
        # Get system status
        credentials_valid = False
        available_models_dict = {}
        
        if bedrock_client:
            credentials_valid = bedrock_client.is_credentials_valid()
            if credentials_valid:
                available_models_dict = bedrock_client.get_available_models()
        
        return render_template('index.html', 
                             credentials_valid=credentials_valid,
                             available_models=available_models_dict,
                             session_id=session_id)
    except Exception as e:
        logger.error(f"Error in chat route: {str(e)}")
        return redirect(url_for('welcome'))

@app.route('/new_chat', methods=['POST'])
def new_chat():
    """Create a new chat"""
    try:
        chat_id = database.create_chat(
            first_message="",  # Empty for manually created chats
            chat_type="Tool-Assisted Research",
            model_name="New Chat"
        )
        
        # Broadcast chat list update
        socketio.emit('chat_list_updated', {
            'chats': database.get_recent_chats(limit=Config.MAX_SIDEBAR_CHATS)
        })
        
        return jsonify({
            'success': True,
            'chat_id': chat_id,
            'message': 'New chat created successfully'
        })
    except Exception as e:
        logger.error(f"Error creating new chat: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/chats')
def get_chats():
    """Get list of recent chats"""
    try:
        chats = database.get_recent_chats(limit=Config.MAX_SIDEBAR_CHATS)
        return jsonify({
            'success': True,
            'chats': chats
        })
    except Exception as e:
        logger.error(f"Error getting chats: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/chats/<int:chat_id>/messages')
def get_chat_messages(chat_id):
    """Get messages for a specific chat"""
    try:
        messages = database.get_chat_messages(chat_id)
        return jsonify({
            'success': True,
            'messages': messages
        })
    except Exception as e:
        logger.error(f"Error getting chat messages: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/chat/<int:chat_id>/delete', methods=['POST'])
def delete_chat(chat_id):
    """Delete a chat"""
    try:
        database.delete_chat(chat_id)
        
        # Broadcast chat list update
        socketio.emit('chat_list_updated', {
            'chats': database.get_recent_chats(limit=Config.MAX_SIDEBAR_CHATS)
        })
        
        return jsonify({
            'success': True,
            'message': 'Chat deleted successfully'
        })
    except Exception as e:
        logger.error(f"Error deleting chat: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/chats/<int:chat_id>', methods=['DELETE'])
def delete_chat_api(chat_id):
    """Delete a chat via API"""
    try:
        database.delete_chat(chat_id)
        
        # Broadcast chat list update
        socketio.emit('chat_list_updated', {
            'chats': database.get_recent_chats(limit=Config.MAX_SIDEBAR_CHATS)
        })
        
        return jsonify({
            'success': True,
            'message': 'Chat deleted successfully'
        })
    except Exception as e:
        logger.error(f"Error deleting chat: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ============================================================================
# WebSocket Events
# ============================================================================

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info(f"Client connected: {request.sid}")
    
    # Send initial chat list
    try:
        chats = database.get_recent_chats(limit=Config.MAX_SIDEBAR_CHATS)
        emit('chat_list_updated', {'chats': chats})
    except Exception as e:
        logger.error(f"Error sending initial chat list: {str(e)}")

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info(f"Client disconnected: {request.sid}")

@socketio.on('send_message')
def handle_message(data):
    """Handle incoming chat messages with streaming support"""
    try:
        message = data.get('message', '').strip()
        chat_type = data.get('chat_type', 'Tool-Assisted Research')
        model_id = data.get('model_id')
        model_name = data.get('model_name', 'Unknown Model')
        frontend_chat_id = data.get('chat_id')  # Chat ID from frontend
        
        if not message:
            emit('error', {'message': 'Message cannot be empty'})
            return
        
        if not model_id:
            emit('error', {'message': 'Please select a model'})
            return
        
        if not chat_agent:
            emit('error', {'message': 'Chat agent not initialized. Please check your AWS credentials.'})
            return
        
        # Use chat ID from frontend or create new chat
        chat_id = frontend_chat_id
        if not chat_id:
            # Create new chat only if no chat ID provided
            chat_id = database.create_chat(
                first_message=message,  # Use the actual user message for naming
                chat_type=chat_type,
                model_name=model_name
            )
            
            # Notify frontend of new chat ID
            emit('chat_created', {'chat_id': chat_id})
            
            # Broadcast chat list update
            socketio.emit('chat_list_updated', {
                'chats': database.get_recent_chats(limit=Config.MAX_SIDEBAR_CHATS)
            })
        
        # Save user message
        database.add_message(
            chat_id=chat_id,
            role='user',
            content=message
        )
        
        # Don't emit user message immediately - let the frontend handle it
        # The message will be displayed when the chat reloads or via other mechanisms
        
        # Start async response generation
        socketio.start_background_task(generate_streaming_response, message, chat_id, chat_type, model_id)
        
    except Exception as e:
        logger.error(f"Error handling message: {str(e)}")
        emit('error', {'message': f'Error processing message: {str(e)}'})

def generate_streaming_response(user_input: str, chat_id: int, chat_type: str, model_id: str):
    """Generate streaming AI response"""
    try:
        # Get chat history
        messages = database.get_chat_messages(chat_id)
        
        # Convert to LangChain format
        langchain_messages = []
        for msg in messages:
            if msg['role'] == 'user':
                langchain_messages.append(HumanMessage(content=msg['content']))
            elif msg['role'] == 'assistant':
                langchain_messages.append(AIMessage(content=msg['content']))
        
        # Prepare state for the agent
        state = {
            'messages': langchain_messages,
            'chat_type': chat_type,
            'model_id': model_id,
            'total_messages': len(langchain_messages),
            'context_messages': 0,
            'estimated_tokens': 0,
            'cache_used': False,
            'iteration_count': 0,
            'max_iterations': Config.MAX_TOOL_ITERATIONS,
            'tools_used': [],
            'tool_results': [],
            'current_thought': '',
            'next_action': '',
            'action_input': '',
            'observation': '',
            'final_answer': ''
        }
        
        # Generate response using the agent with streaming
        config = {"configurable": {"thread_id": f"chat_{chat_id}"}}
        
        # Collect metadata during streaming
        metadata = {
            'thinking_steps': [],
            'tool_calls': [],
            'reasoning': ''
        }
        
        # Emit thinking indicator
        socketio.emit('thinking', {'thinking': True, 'step': 'Starting to think...'})
        metadata['thinking_steps'].append('Starting to think...')
        
        # Run the async stream_agent_response in a proper async loop
        async def run_streaming():
            return await stream_agent_response(state, config, chat_id, metadata)
        
        # Execute the async function properly
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(run_streaming())
        finally:
            loop.close()
        
        # Stop thinking indicator
        socketio.emit('thinking', {'thinking': False})
        
        # Save assistant message with metadata
        final_response = result.get('final_answer', 'Unable to generate response')
        database.add_message(
            chat_id=chat_id,
            role='assistant',
            content=final_response,
            metadata=json.dumps(metadata) if metadata['thinking_steps'] or metadata['tool_calls'] or metadata['reasoning'] else None
        )
        
        # Update chat title if this is the first assistant response
        # Check if this chat only has the user message (before we added the assistant response)
        chat_message_count = database.get_chat_message_count(chat_id)
        if chat_message_count == 2:  # User message + assistant response = first complete exchange
            # Generate a better title using the first user message
            title = database.generate_smart_chat_title(user_input)
            database.update_chat_title(chat_id, title)
            
            # Emit updated chat list
            socketio.emit('chat_list_updated', {
                'chats': database.get_recent_chats(limit=Config.MAX_SIDEBAR_CHATS)
            })
        
        # Update chat list
        socketio.emit('chat_list_updated', {
            'chats': database.get_recent_chats(limit=Config.MAX_SIDEBAR_CHATS)
        })
        
    except Exception as e:
        logger.error(f"Error generating streaming response: {str(e)}")
        socketio.emit('error', {'message': f'Error generating response: {str(e)}'})
        socketio.emit('thinking', {'thinking': False})

async def stream_agent_response(state, config, chat_id, metadata):
    """Stream agent response using REAL-TIME streaming for thinking, action planning, and responses"""
    try:
        socketio.emit('thinking', {'thinking': True, 'step': 'Starting ReAct workflow...'})
        metadata['thinking_steps'].append('Starting ReAct workflow...')
        
        full_response = ""
        
        # Phase 1: REAL-TIME Reasoning Streaming
        socketio.emit('thinking', {'thinking': True, 'step': 'Reasoning about your request...'})
        reasoning_text = ""
        
        async for reasoning_chunk in chat_agent._stream_reasoning_node(state):
            reasoning_text += reasoning_chunk
            # Emit each thinking chunk in real-time
            socketio.emit('thinking_chunk', {
                'chunk': reasoning_chunk,
                'complete': False,
                'type': 'reasoning'
            })
            await asyncio.sleep(0.01)  # Small delay for smooth streaming
        
        # Complete reasoning phase
        socketio.emit('thinking_chunk', {
            'chunk': '',
            'complete': True,
            'type': 'reasoning'
        })
        metadata['reasoning'] = reasoning_text
        
        # Phase 2: REAL-TIME Action Planning Streaming
        # Always run action planning - let the agent decide if tools are needed
        socketio.emit('thinking', {'thinking': True, 'step': 'Planning action...'})
        planning_text = ""
        
        async for planning_chunk in chat_agent._stream_action_planning_node(state):
            planning_text += planning_chunk
            # Emit each planning chunk in real-time
            socketio.emit('thinking_chunk', {
                'chunk': planning_chunk,
                'complete': False,
                'type': 'planning'
            })
            await asyncio.sleep(0.01)
        
        # Complete planning phase
        socketio.emit('thinking_chunk', {
            'chunk': '',
            'complete': True,
            'type': 'planning'
        })
        
        # Execute tool if needed (state is updated by the streaming method)
        if state.get('next_action') and state.get('next_action') != 'none':
            tool_call = {
                'tool': state.get('next_action'),
                'input': state.get('action_input', ''),
                'status': 'Running'
            }
            socketio.emit('tool_execution', tool_call)
            metadata['tool_calls'].append(tool_call.copy())
            
            # Execute the tool
            tool_result = await chat_agent._tool_execution_node(state)
            
            if tool_result.get('tool_results'):
                latest_result = tool_result['tool_results'][-1]
                completed_tool = {
                    'tool': latest_result['tool'],
                    'input': latest_result['input'],
                    'output': latest_result['output'],
                    'status': 'Complete'
                }
                socketio.emit('tool_execution', completed_tool)
                metadata['tool_calls'].append(completed_tool)
            
            # Update state with tool results
            state.update(tool_result)
            
            # Observation phase
            socketio.emit('thinking', {'thinking': True, 'step': 'Analyzing tool results...'})
            observation_result = await chat_agent._observation_node(state)
            state.update(observation_result)
        
        # Phase 3: REAL-TIME Response Generation Streaming
        socketio.emit('thinking', {'thinking': False})
        socketio.emit('thinking', {'thinking': True, 'step': 'Generating response...'})
        
        # Stream the final response generation in real-time
        streamed_response = ""
        async for response_chunk in chat_agent._stream_response_node(state):
            streamed_response += response_chunk
            # Emit response chunks in real-time
            socketio.emit('message_chunk', {
                'chunk': response_chunk,
                'complete': False
            })
            await asyncio.sleep(0.01)
        
        # Mark streaming complete
        socketio.emit('message_chunk', {
            'chunk': '',
            'complete': True
        })
        
        full_response = streamed_response.strip()
        
        # Turn off thinking indicator
        socketio.emit('thinking', {'thinking': False})
        
        # Send final formatted message
        socketio.emit('message_final', {
            'role': 'assistant',
            'content': full_response or "I apologize, but I couldn't generate a response.",
            'metadata': metadata if metadata['thinking_steps'] or metadata['tool_calls'] else None,
            'timestamp': datetime.now().isoformat()
        })
        
        return {'final_answer': full_response}
        
    except Exception as e:
        logger.error(f"Error in real-time streaming: {str(e)}")
        socketio.emit('error', {'message': f'Error in agent response: {str(e)}'})
        socketio.emit('thinking', {'thinking': False})
        return {'final_answer': f'Error: {str(e)}'}

# ============================================================================
# Error Handlers
# ============================================================================

@app.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors"""
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {str(error)}")
    return render_template('500.html'), 500

# ============================================================================
# Main Application Entry Point
# ============================================================================

if __name__ == '__main__':
    try:
        # Validate configuration
        config_errors = Config.validate_config()
        if config_errors:
            logger.error("Configuration validation failed:")
            for error in config_errors:
                logger.error(f"  - {error}")
            exit(1)
        
        # Start the application
        logger.info("Starting Bedrock Chat Application...")
        logger.info(f"Environment: {Config.get_environment_info()}")
        
        # Run with SocketIO
        socketio.run(
            app,
            host='0.0.0.0',
            port=5000,
            debug=False,
            allow_unsafe_werkzeug=True
        )
        
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        exit(1) 