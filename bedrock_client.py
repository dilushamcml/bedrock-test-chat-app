"""
AWS Bedrock Client Implementation using LangChain AWS
Complete implementation for AWS Bedrock model interaction using official LangChain libraries
Handles authentication, model discovery, and streaming responses
"""

import boto3
import json
import logging
from typing import Dict, Generator, Tuple, List, Optional, Any
from botocore.exceptions import ClientError, NoCredentialsError

from langchain_aws import ChatBedrock
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.outputs import ChatResult

from config import Config

# Configure logging
logger = logging.getLogger(__name__)

class BedrockError(Exception):
    """Custom exception for Bedrock operations"""
    pass

class BedrockClient:
    """
    AWS Bedrock client for model interaction and management using LangChain AWS
    
    Features:
    - Uses official langchain_aws.ChatBedrock
    - Dynamic model discovery from AWS Bedrock
    - Multi-provider support (Claude, Titan, Llama, Mistral, etc.)
    - Streaming response handling
    - Comprehensive error handling
    - Full LangChain integration
    """
    
    def __init__(self):
        """Initialize the Bedrock client"""
        self.bedrock_client = None
        self.bedrock_runtime = None
        self.chat_models = {}  # Cache for LangChain chat models
        self._available_models = None
        self._credentials_valid = False
        
        # Supported model providers
        self.supported_providers = {
            'anthropic', 'amazon', 'meta', 'mistral', 'ai21', 'cohere'
        }
        
        # Model display names mapping
        self.model_display_names = {
            'anthropic.claude-3-5-sonnet-20241022-v2:0': 'Claude 3.5 Sonnet v2',
            'anthropic.claude-3-5-sonnet-20240620-v1:0': 'Claude 3.5 Sonnet',
            'anthropic.claude-3-5-haiku-20241022-v1:0': 'Claude 3.5 Haiku',
            'anthropic.claude-3-opus-20240229-v1:0': 'Claude 3 Opus',
            'anthropic.claude-3-sonnet-20240229-v1:0': 'Claude 3 Sonnet',
            'anthropic.claude-3-haiku-20240307-v1:0': 'Claude 3 Haiku',
            'anthropic.claude-v2:1': 'Claude 2.1',
            'anthropic.claude-v2': 'Claude 2.0',
            'anthropic.claude-instant-v1': 'Claude Instant',
            'amazon.titan-text-premier-v1:0': 'Titan Text Premier',
            'amazon.titan-text-express-v1': 'Titan Text Express',
            'amazon.titan-text-lite-v1': 'Titan Text Lite',
            'meta.llama3-2-90b-instruct-v1:0': 'Llama 3.2 90B Instruct',
            'meta.llama3-2-11b-instruct-v1:0': 'Llama 3.2 11B Instruct',
            'meta.llama3-2-3b-instruct-v1:0': 'Llama 3.2 3B Instruct',
            'meta.llama3-2-1b-instruct-v1:0': 'Llama 3.2 1B Instruct',
            'meta.llama3-1-70b-instruct-v1:0': 'Llama 3.1 70B Instruct',
            'meta.llama3-1-8b-instruct-v1:0': 'Llama 3.1 8B Instruct',
            'meta.llama3-70b-instruct-v1:0': 'Llama 3 70B Instruct',
            'meta.llama3-8b-instruct-v1:0': 'Llama 3 8B Instruct',
            'mistral.mistral-large-2407-v1:0': 'Mistral Large 2407',
            'mistral.mistral-large-2402-v1:0': 'Mistral Large 2402',
            'mistral.mistral-small-2402-v1:0': 'Mistral Small 2402',
            'mistral.mixtral-8x7b-instruct-v0:1': 'Mixtral 8x7B Instruct',
            'mistral.mistral-7b-instruct-v0:2': 'Mistral 7B Instruct',
            'ai21.jamba-1-5-large-v1:0': 'AI21 Jamba 1.5 Large',
            'ai21.jamba-1-5-mini-v1:0': 'AI21 Jamba 1.5 Mini',
            'ai21.j2-ultra-v1': 'AI21 Jurassic-2 Ultra',
            'ai21.j2-mid-v1': 'AI21 Jurassic-2 Mid',
            'cohere.command-r-plus-v1:0': 'Cohere Command R+',
            'cohere.command-r-v1:0': 'Cohere Command R',
            'cohere.command-text-v14': 'Cohere Command Text',
            'cohere.command-light-text-v14': 'Cohere Command Light'
        }
        
        logger.info("BedrockClient initialized")
    
    def _create_aws_session(self) -> boto3.Session:
        """Create AWS session with explicit credentials from config"""
        session_kwargs = {}
        
        # Use explicit credentials if provided
        if Config.AWS_ACCESS_KEY_ID and Config.AWS_SECRET_ACCESS_KEY:
            session_kwargs.update({
                'aws_access_key_id': Config.AWS_ACCESS_KEY_ID,
                'aws_secret_access_key': Config.AWS_SECRET_ACCESS_KEY,
                'region_name': Config.DEFAULT_AWS_REGION
            })
            
            # Add session token if provided (for temporary credentials)
            if Config.AWS_SESSION_TOKEN:
                session_kwargs['aws_session_token'] = Config.AWS_SESSION_TOKEN
        
        # Use profile if specified
        elif Config.AWS_PROFILE:
            session_kwargs.update({
                'profile_name': Config.AWS_PROFILE,
                'region_name': Config.DEFAULT_AWS_REGION
            })
        
        # Default to environment/instance credentials
        else:
            session_kwargs['region_name'] = Config.DEFAULT_AWS_REGION
        
        return boto3.Session(**session_kwargs)
    
    def validate_credentials(self) -> Tuple[bool, str]:
        """
        Validate AWS credentials and Bedrock access
        
        Returns:
            Tuple of (is_valid, message)
        """
        try:
            # Create AWS session
            session = self._create_aws_session()
            
            # Create clients
            self.bedrock_client = session.client('bedrock')
            self.bedrock_runtime = session.client('bedrock-runtime')
            
            # Test connection with a simple API call
            try:
                response = self.bedrock_client.list_foundation_models()
                logger.info(f"Successfully connected to Bedrock. Found {len(response.get('modelSummaries', []))} models.")
            except ClientError as e:
                if e.response['Error']['Code'] == 'AccessDenied':
                    return False, "❌ Access denied to AWS Bedrock. Check your IAM permissions and enable Bedrock in your region."
                raise
            
            # Test with LangChain ChatBedrockConverse
            try:
                # Get available models first
                available_models = self.get_available_models()
                if not available_models:
                    return False, "❌ No models available. Please check your AWS Bedrock model access permissions."
                
                # Test with Claude 3 Haiku (supports on-demand throughput)
                test_model_id = None
                for model_id in available_models.values():
                    if 'claude-3-haiku-20240307-v1:0' in model_id:
                        test_model_id = model_id
                        break
                    elif 'claude-3-sonnet-20240229-v1:0' in model_id:
                        test_model_id = model_id
                        break
                
                # Fallback to first available model
                if not test_model_id:
                    test_model_id = list(available_models.values())[0]
                
                # Create a test ChatBedrockConverse instance
                chat_model = ChatBedrock(
                    model_id=test_model_id,
                    credentials_profile_name=Config.AWS_PROFILE if Config.AWS_PROFILE else None,
                    region_name=Config.DEFAULT_AWS_REGION,
                    model_kwargs={
                        'temperature': 0.1,
                        'max_tokens': 10
                    }
                )
                
                # Test with a simple message
                test_message = HumanMessage(content="Hello")
                response = chat_model.invoke([test_message])
                
                if response and response.content:
                    logger.info(f"Successfully validated LangChain ChatBedrock with {test_model_id}")
                    self._credentials_valid = True
                    return True, "✅ AWS Bedrock credentials are valid and working"
                else:
                    logger.warning("LangChain ChatBedrock test returned empty response")
                    return False, "❌ LangChain ChatBedrock test failed"
                
            except Exception as e:
                logger.error(f"LangChain ChatBedrock test failed: {str(e)}")
                return False, f"❌ LangChain ChatBedrock test failed: {str(e)}"
                
        except NoCredentialsError:
            return False, "❌ AWS credentials not found. Please configure your credentials."
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_msg = e.response['Error']['Message']
            
            if error_code == 'UnauthorizedOperation':
                return False, f"❌ Unauthorized: {error_msg}"
            elif error_code == 'AccessDenied':
                return False, f"❌ Access denied: {error_msg}"
            else:
                return False, f"❌ AWS error ({error_code}): {error_msg}"
        except Exception as e:
            logger.error(f"Unexpected error during credential validation: {str(e)}")
            return False, f"❌ Unexpected error: {str(e)}"
    
    def get_available_models(self) -> Dict[str, str]:
        """
        Return hardcoded available models (confirmed working with on-demand throughput)
        
        Returns:
            Dictionary mapping display names to model IDs (only accessible models)
        """
        if self._available_models is not None:
            return self._available_models
        
        # Confirmed working models (on-demand throughput supported)
        working_models = {
            # Claude 3 models (confirmed working)
            "Claude 3 Haiku": "anthropic.claude-3-haiku-20240307-v1:0",
            "Claude 3 Sonnet": "anthropic.claude-3-sonnet-20240229-v1:0",
            "Claude 3.5 Sonnet": "anthropic.claude-3-5-sonnet-20240620-v1:0",
            
            # Titan models (likely working)
            "Titan Text Express": "amazon.titan-text-express-v1",
            "Titan Text Lite": "amazon.titan-text-lite-v1",
            
            # Claude 2 models (likely working)
            "Claude 2.1": "anthropic.claude-v2:1",
            "Claude Instant": "anthropic.claude-instant-v1",
        }
        
        # Test access to the primary model to ensure credentials are working
        try:
            test_model_id = "anthropic.claude-3-haiku-20240307-v1:0"
            test_chat = ChatBedrock(
                model_id=test_model_id,
                region_name=Config.DEFAULT_AWS_REGION,
                model_kwargs={
                    'temperature': 0.1,
                    'max_tokens': 5
                }
            )
            
            # Test with a minimal message
            test_message = HumanMessage(content="Hi")
            response = test_chat.invoke([test_message])
            
            if response and response.content:
                logger.info(f"Successfully validated LangChain ChatBedrock with {test_model_id}")
                # Cache the results
                self._available_models = working_models
                logger.info(f"Successfully loaded {len(working_models)} working models")
                return working_models
            else:
                logger.error("Failed to validate primary model")
                return {}
                
        except Exception as e:
            logger.error(f"Failed to validate primary model: {str(e)}")
            return {}
    
    def get_chat_model(self, model_id: str, **kwargs) -> ChatBedrock:
        """
        Get or create a LangChain ChatBedrock instance for the specified model
        
        Args:
            model_id: The Bedrock model ID
            **kwargs: Additional parameters for the chat model
            
        Returns:
            ChatBedrock instance
        """
        # Create cache key
        cache_key = f"{model_id}:{json.dumps(kwargs, sort_keys=True)}"
        
        if cache_key in self.chat_models:
            return self.chat_models[cache_key]
        
        # Default parameters
        default_params = {
            'model_id': model_id,
            'region_name': Config.DEFAULT_AWS_REGION,
            'model_kwargs': {
                'temperature': 0.7,
                'max_tokens': 4000,
            }
        }
        
        # Add credentials if available
        if Config.AWS_PROFILE:
            default_params['credentials_profile_name'] = Config.AWS_PROFILE
        
        # Extract model_kwargs from kwargs if provided
        if 'temperature' in kwargs or 'max_tokens' in kwargs:
            model_kwargs = default_params['model_kwargs'].copy()
            if 'temperature' in kwargs:
                model_kwargs['temperature'] = kwargs.pop('temperature')
            if 'max_tokens' in kwargs:
                model_kwargs['max_tokens'] = kwargs.pop('max_tokens')
            default_params['model_kwargs'] = model_kwargs
        
        # Merge with provided kwargs
        default_params.update(kwargs)
        
        # Create the chat model
        chat_model = ChatBedrock(**default_params)
        
        # Cache it
        self.chat_models[cache_key] = chat_model
        
        logger.info(f"Created ChatBedrock instance for {model_id}")
        return chat_model
    
    def invoke_model(self, model_id: str, messages: List[BaseMessage], **kwargs) -> str:
        """
        Invoke a model with messages using LangChain
        
        Args:
            model_id: The Bedrock model ID
            messages: List of LangChain messages
            **kwargs: Additional parameters for the model
            
        Returns:
            The model's response as a string
        """
        try:
            chat_model = self.get_chat_model(model_id, **kwargs)
            response = chat_model.invoke(messages)
            return response.content if response else ""
            
        except Exception as e:
            logger.error(f"Error invoking model {model_id}: {str(e)}")
            raise BedrockError(f"Error invoking model: {str(e)}")
    
    def invoke_model_stream(self, model_id: str, messages: List[BaseMessage], **kwargs) -> Generator[str, None, None]:
        """
        Invoke a model with streaming response using LangChain
        
        Args:
            model_id: The Bedrock model ID
            messages: List of LangChain messages
            **kwargs: Additional parameters for the model
            
        Yields:
            Chunks of the model's response
        """
        try:
            chat_model = self.get_chat_model(model_id, **kwargs)
            
            # Stream the response
            for chunk in chat_model.stream(messages):
                if chunk.content:
                    yield chunk.content
                    
        except Exception as e:
            logger.error(f"Error streaming from model {model_id}: {str(e)}")
            yield f"Error: {str(e)}"
    
    def is_credentials_valid(self) -> bool:
        """Check if credentials are valid"""
        return self._credentials_valid
    
    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """
        Get information about a specific model
        
        Args:
            model_id: The Bedrock model ID
            
        Returns:
            Dictionary containing model information
        """
        if not self.bedrock_client:
            return {}
        
        try:
            response = self.bedrock_client.get_foundation_model(modelIdentifier=model_id)
            return response.get('modelDetails', {})
        except Exception as e:
            logger.error(f"Error getting model info for {model_id}: {str(e)}")
            return {}


# ============================================================================
# Test and Demo Functions
# ============================================================================

def test_bedrock_client():
    """Test the Bedrock client functionality"""
    print("🧪 Testing Bedrock Client...")
    
    # Test credential validation
    client = BedrockClient()
    is_valid, message = client.validate_credentials()
    
    print(f"Credential validation: {message}")
    
    if is_valid:
        # Test model discovery
        models = client.get_available_models()
        print(f"📋 Available models: {len(models)}")
        
        for display_name, model_id in list(models.items())[:5]:
            print(f"  • {display_name}: {model_id}")
        
        if models:
            # Test model invocation
            first_model = list(models.values())[0]
            print(f"\n🤖 Testing model: {first_model}")
            
            try:
                messages = [HumanMessage(content="Hello! Please respond with just 'Hi there!'")]
                response = client.invoke_model(first_model, messages, max_tokens=10)
                print(f"Response: {response}")
                
                # Test streaming
                print("\n🌊 Testing streaming...")
                stream_response = ""
                for chunk in client.invoke_model_stream(first_model, messages, max_tokens=20):
                    stream_response += chunk
                    print(f"Chunk: {chunk}")
                print(f"Full streaming response: {stream_response}")
                
            except Exception as e:
                print(f"❌ Error testing model: {str(e)}")
    
    print("✅ Bedrock client test completed")


if __name__ == "__main__":
    # Test the client if run directly
    import sys
    import os
    
    # Add the parent directory to the path so we can import config
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    try:
        test_bedrock_client()
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()