"""
Configuration settings for the Bedrock Chat Application
Centralized configuration management with environment variable support
"""

import os
import logging
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Application configuration settings"""
    
    # ============================================================================
    # AWS Configuration
    # ============================================================================
    
    # AWS Region Configuration
    DEFAULT_AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    
    # AWS Profile (optional)
    AWS_PROFILE = os.getenv("AWS_PROFILE")
    
    # AWS Credentials (required for Bedrock access)
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_SESSION_TOKEN = os.getenv("AWS_SESSION_TOKEN")  # Optional for temporary credentials
    
    # ============================================================================
    # Bedrock Configuration
    # ============================================================================
    
    # Bedrock model preferences (in order of preference)
    PREFERRED_MODELS = [
        "anthropic.claude-3-5-sonnet-20241022-v2:0",  # Claude 3.5 Sonnet v2
        "anthropic.claude-3-5-sonnet-20240620-v1:0",  # Claude 3.5 Sonnet
        "anthropic.claude-3-sonnet-20240229-v1:0",    # Claude 3 Sonnet
        "anthropic.claude-3-haiku-20240307-v1:0",     # Claude 3 Haiku
        "amazon.titan-text-express-v1",               # Titan Text Express
    ]
    
    # Default model ID (will be set dynamically based on available models)
    DEFAULT_MODEL_ID = None
    
    # Streaming configuration
    ENABLE_STREAMING = True
    STREAMING_CHUNK_SIZE = 50  # Characters per chunk for UI streaming
    
    # Model validation settings
    VALIDATE_MODELS_ON_STARTUP = True
    MODEL_VALIDATION_TIMEOUT = 30  # seconds
    
    # ============================================================================
    # Database Configuration
    # ============================================================================
    
    # Database path (SQLite)
    DATABASE_PATH = os.getenv("DATABASE_PATH", "chat_database.db")
    
    @staticmethod
    def get_database_path() -> Path:
        """Get the database path as a Path object"""
        return Path(Config.DATABASE_PATH)
    
    # ============================================================================
    # Context Management Configuration
    # ============================================================================
    
    # Maximum number of messages to keep in conversation context
    MAX_CONTEXT_MESSAGES = int(os.getenv("MAX_CONTEXT_MESSAGES", "10"))
    
    # Maximum number of tokens to keep in context
    MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "8000"))
    
    # Token estimation multiplier for safety margin
    TOKEN_SAFETY_MULTIPLIER = float(os.getenv("TOKEN_SAFETY_MULTIPLIER", "1.2"))
    
    # ============================================================================
    # Chat Configuration
    # ============================================================================
    
    # Available chat types
    CHAT_TYPES = [
        "General Chat",
        "Code Assistant", 
        "Creative Writing",
        "Analysis & Research",
        "Question & Answer",
        "Tool-Assisted Research"  # New type for tool-enabled conversations
    ]
    
    # Default chat type
    DEFAULT_CHAT_TYPE = "General Chat"
    
    # ============================================================================
    # Tool Configuration
    # ============================================================================
    
    # Enable tool calling
    ENABLE_TOOLS = os.getenv("ENABLE_TOOLS", "true").lower() == "true"
    
    # Available tools
    AVAILABLE_TOOLS = [
        "web_search",
        "calculator", 
        "code_executor",
        "file_reader",
        "weather_api",
        "database_query"
    ]
    
    # Tool-specific settings
    WEB_SEARCH_API_KEY = os.getenv("WEB_SEARCH_API_KEY")  # For real web search
    WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
    
    # Tool execution limits
    MAX_TOOL_ITERATIONS = int(os.getenv("MAX_TOOL_ITERATIONS", "5"))
    TOOL_TIMEOUT = int(os.getenv("TOOL_TIMEOUT", "30"))  # seconds
    
    # ============================================================================
    # Streamlit Configuration
    # ============================================================================
    
    # Flask page configuration
    PAGE_TITLE = "🤖 Bedrock Chat Assistant"
    PAGE_ICON = "🤖"
    
    # ============================================================================
    # Logging Configuration
    # ============================================================================
    
    # Log level
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
    
    # Log format
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Log file path (optional)
    LOG_FILE = os.getenv("LOG_FILE")
    
    @staticmethod
    def setup_logging():
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, Config.LOG_LEVEL),
            format=Config.LOG_FORMAT,
            handlers=[
                logging.StreamHandler(),
                *([logging.FileHandler(Config.LOG_FILE)] if Config.LOG_FILE else [])
            ]
        )
    
    # ============================================================================
    # Application Settings
    # ============================================================================
    
    # Environment (development, production)
    ENVIRONMENT = os.getenv("ENVIRONMENT", "production")
    
    # Debug mode
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    
    # Application version
    VERSION = "0.2.0"  # Updated version with tools
    
    # ============================================================================
    # UI Configuration
    # ============================================================================
    
    # Maximum number of chats to show in sidebar
    MAX_SIDEBAR_CHATS = int(os.getenv("MAX_SIDEBAR_CHATS", "20"))
    
    # Chat name length limit
    MAX_CHAT_NAME_LENGTH = int(os.getenv("MAX_CHAT_NAME_LENGTH", "50"))
    
    # Message display configuration
    MAX_MESSAGE_DISPLAY_LENGTH = int(os.getenv("MAX_MESSAGE_DISPLAY_LENGTH", "1000"))
    
    # Auto-scroll to bottom
    AUTO_SCROLL = os.getenv("AUTO_SCROLL", "true").lower() == "true"
    
    # ============================================================================
    # Performance Configuration
    # ============================================================================
    
    # Flask caching settings
    CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))  # 1 hour
    
    # Database connection timeout
    DB_TIMEOUT = int(os.getenv("DB_TIMEOUT", "30"))  # 30 seconds
    
    # Request timeout for Bedrock API calls
    BEDROCK_TIMEOUT = int(os.getenv("BEDROCK_TIMEOUT", "60"))  # 60 seconds
    
    # ============================================================================
    # Feature Flags
    # ============================================================================
    
    # Enable/disable features
    ENABLE_CHAT_SEARCH = os.getenv("ENABLE_CHAT_SEARCH", "true").lower() == "true"
    ENABLE_EXPORT_CHAT = os.getenv("ENABLE_EXPORT_CHAT", "true").lower() == "true"
    ENABLE_CHAT_STATISTICS = os.getenv("ENABLE_CHAT_STATISTICS", "true").lower() == "true"
    ENABLE_DEBUG_PANEL = os.getenv("ENABLE_DEBUG_PANEL", "true").lower() == "true"
    
    # ============================================================================
    # Validation Methods
    # ============================================================================
    
    @staticmethod
    def validate_config() -> List[str]:
        """
        Validate configuration settings
        
        Returns:
            List of validation errors (empty if all valid)
        """
        errors = []
        
        # Validate AWS credentials
        if not Config.AWS_ACCESS_KEY_ID:
            errors.append("AWS_ACCESS_KEY_ID is required for Bedrock access")
        
        if not Config.AWS_SECRET_ACCESS_KEY:
            errors.append("AWS_SECRET_ACCESS_KEY is required for Bedrock access")
        
        # Validate context limits
        if Config.MAX_CONTEXT_MESSAGES < 1:
            errors.append("MAX_CONTEXT_MESSAGES must be at least 1")
        
        if Config.MAX_CONTEXT_TOKENS < 100:
            errors.append("MAX_CONTEXT_TOKENS must be at least 100")
        
        # Validate chat types
        if not Config.CHAT_TYPES:
            errors.append("CHAT_TYPES cannot be empty")
        
        if Config.DEFAULT_CHAT_TYPE not in Config.CHAT_TYPES:
            errors.append(f"DEFAULT_CHAT_TYPE '{Config.DEFAULT_CHAT_TYPE}' not in CHAT_TYPES")
        
        # Validate UI limits
        if Config.MAX_SIDEBAR_CHATS < 1:
            errors.append("MAX_SIDEBAR_CHATS must be at least 1")
        
        if Config.MAX_CHAT_NAME_LENGTH < 10:
            errors.append("MAX_CHAT_NAME_LENGTH must be at least 10")
        
        # Validate timeouts
        if Config.DB_TIMEOUT < 1:
            errors.append("DB_TIMEOUT must be at least 1 second")
        
        if Config.BEDROCK_TIMEOUT < 1:
            errors.append("BEDROCK_TIMEOUT must be at least 1 second")
        
        # Validate tool configuration
        if Config.ENABLE_TOOLS and Config.MAX_TOOL_ITERATIONS < 1:
            errors.append("MAX_TOOL_ITERATIONS must be at least 1 when tools are enabled")
        
        return errors
    
    @staticmethod
    def get_environment_info() -> dict:
        """Get environment information for debugging"""
        return {
            "environment": Config.ENVIRONMENT,
            "debug": Config.DEBUG,
            "log_level": Config.LOG_LEVEL,
            "aws_region": Config.DEFAULT_AWS_REGION,
            "aws_profile": Config.AWS_PROFILE,
            "aws_credentials_configured": bool(Config.AWS_ACCESS_KEY_ID and Config.AWS_SECRET_ACCESS_KEY),
            "database_path": str(Config.get_database_path()),
            "max_context_messages": Config.MAX_CONTEXT_MESSAGES,
            "max_context_tokens": Config.MAX_CONTEXT_TOKENS,
            "tools_enabled": Config.ENABLE_TOOLS,
            "streaming_enabled": Config.ENABLE_STREAMING,
            "version": Config.VERSION
        }
    
    @staticmethod
    def get_aws_credentials() -> dict:
        """Get AWS credentials for boto3 session"""
        credentials = {}
        
        if Config.AWS_ACCESS_KEY_ID:
            credentials['aws_access_key_id'] = Config.AWS_ACCESS_KEY_ID
        
        if Config.AWS_SECRET_ACCESS_KEY:
            credentials['aws_secret_access_key'] = Config.AWS_SECRET_ACCESS_KEY
        
        if Config.AWS_SESSION_TOKEN:
            credentials['aws_session_token'] = Config.AWS_SESSION_TOKEN
        
        if Config.DEFAULT_AWS_REGION:
            credentials['region_name'] = Config.DEFAULT_AWS_REGION
        
        return credentials

# Initialize logging when module is imported
Config.setup_logging()

# Validate configuration on import
config_errors = Config.validate_config()
if config_errors:
    logger = logging.getLogger(__name__)
    logger.warning(f"Configuration validation errors: {config_errors}")
    if not Config.DEBUG:
        # In production, we want to fail fast if credentials are missing
        critical_errors = [error for error in config_errors if "AWS_" in error]
        if critical_errors:
            logger.error("Critical AWS credential errors found. Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables.")
            logger.error("You can also use AWS profiles or other credential methods supported by boto3.") 