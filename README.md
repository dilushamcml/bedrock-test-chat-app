# 🤖 Bedrock Chat Assistant

A simple, powerful chat application using AWS Bedrock models with LangGraph for intelligent context management and SQLite for persistent storage.

## 🚀 Quick Start with Poetry

### Prerequisites
- Python 3.9 or higher
- [Poetry](https://python-poetry.org/docs/#installation) installed
- AWS account with Bedrock access

### 1. Clone and Setup
```bash
# Clone the repository
git clone <your-repo-url>
cd bedrock-chat-app

# Install dependencies with Poetry
poetry install

# Activate the virtual environment
poetry shell
```

### 2. Configure AWS
Choose one of these methods:

#### Option A: AWS CLI (Recommended)
```bash
aws configure
```

#### Option B: Environment Variables
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your credentials
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-east-1
```

#### Option C: AWS Profile
```bash
export AWS_PROFILE=your_profile_name
```

### 3. Enable Bedrock Models
1. Open [AWS Bedrock Console](https://console.aws.amazon.com/bedrock/)
2. Go to "Model access" in the left sidebar
3. Request access to desired models (Claude, Titan, etc.)
4. Wait for approval (usually instant)

### 4. Run the Application
```bash
# Using Poetry
poetry run streamlit run main.py

# Or if in Poetry shell
streamlit run main.py
```

The application will open in your browser at `http://localhost:8501`

## 📁 Project Structure

```
bedrock-chat-app/
├── pyproject.toml          # Poetry configuration & dependencies
├── poetry.lock             # Poetry lock file (auto-generated)
├── .env.example            # Environment template
├── config.py               # Application configuration
├── database.py             # SQLite database operations
├── bedrock_client.py       # AWS Bedrock client (separate file)
├── chat_agent.py          # LangGraph agent with context management
├── main.py                # Streamlit main application
├── README.md              # This file
└── chat_database.db       # SQLite database (auto-created)
```

## 🔢 Implementation Order

If building from scratch, implement in this order:

1. **`pyproject.toml`** - Poetry configuration
2. **`config.py`** - Application settings
3. **`.env.example`** - Environment template
4. **`database.py`** - SQLite operations (no dependencies)
5. **`bedrock_client.py`** - AWS Bedrock client (depends on config)
6. **`chat_agent.py`** - LangGraph agent (depends on bedrock_client, config)
7. **`main.py`** - Streamlit app (depends on all components)
8. **`README.md`** - Documentation

## 🎯 Features

### ✅ **Simple & Clean**
- Streamlit-based UI with intuitive chat interface
- Sidebar navigation and chat management
- Real-time message streaming

### ✅ **LangGraph Integration**
- Context management with message trimming
- State persistence across sessions
- Simple but effective design following LangGraph patterns

### ✅ **AWS Bedrock Support**
- **Separate client module** as requested
- Dynamic model discovery from your AWS account
- Support for Claude, Titan, Llama, Mistral models
- Proper error handling and credential validation

### ✅ **SQLite Persistence**
- Chat and message storage
- Search functionality
- Database statistics and management

### ✅ **Context Management**
- Simple message trimming (keeps last 10 messages)
- Configurable context limits
- Token estimation and debugging

## ⚙️ Configuration

### Core Settings (`config.py`)
```python
# Context Management
MAX_CONTEXT_MESSAGES = 10  # Messages to keep in context
MAX_CONTEXT_TOKENS = 8000  # Token limit

# Chat Types
CHAT_TYPES = [
    "General Chat",
    "Code Assistant", 
    "Creative Writing",
    "Analysis & Research",
    "Question & Answer"
]
```

### Environment Variables (`.env`)
```bash
# AWS Configuration
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=us-east-1

# Application Settings
DATABASE_PATH=chat_database.db
LOG_LEVEL=INFO
```

## 🧠 LangGraph Implementation

This application follows LangGraph best practices:

1. **StateGraph** - Manages conversation state with proper schema
2. **Message Trimming** - Keeps context within limits (no summarization)
3. **Memory Persistence** - Uses MemorySaver for thread persistence
4. **Simple Flow** - START → chatbot → END

### Context Management Strategy
- Keeps last **10 messages** in context
- Ensures conversation starts with user message
- Token-aware trimming as secondary constraint
- Debug panel shows context details

## 🔧 Development

### Using Poetry for Development

```bash
# Add new dependencies
poetry add package_name

# Add development dependencies
poetry add --group dev package_name

# Update dependencies
poetry update

# Run tests (when added)
poetry run pytest

# Check code formatting
poetry run black .
poetry run flake8
```

### Development Mode
Set environment variable for detailed error messages:
```bash
export ENVIRONMENT=development
```

## 🐛 Troubleshooting

### Common Issues

#### 1. **Poetry Installation**
```bash
# Install Poetry if not installed
curl -sSL https://install.python-poetry.org | python3 -
```

#### 2. **AWS Credentials**
```bash
# Check AWS configuration
aws sts get-caller-identity

# Verify Bedrock access
aws bedrock list-foundation-models --region us-east-1
```

#### 3. **Model Access Denied**
- Check AWS Bedrock Console → Model Access
- Ensure models are enabled in your region
- Verify IAM permissions for Bedrock

#### 4. **Database Issues**
```bash
# Reset database
rm chat_database.db
# App will recreate it automatically
```

#### 5. **Streamlit Cache Issues**
```bash
poetry run streamlit cache clear
```

### Debug Mode
Enable debug information in the sidebar to see:
- Total messages in conversation
- Context messages sent to LLM
- Token usage estimates
- Context management strategy

## 📊 Database Schema

### Chats Table
```sql
CREATE TABLE chats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    chat_type TEXT DEFAULT 'General Chat',
    model_name TEXT DEFAULT '',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    message_count INTEGER DEFAULT 0,
    is_active BOOLEAN DEFAULT 1
);
```

### Messages Table
```sql
CREATE TABLE messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    token_count INTEGER DEFAULT 0,
    FOREIGN KEY (chat_id) REFERENCES chats (id) ON DELETE CASCADE
);
```

## 🚀 Deployment

### Local Deployment
```bash
poetry run streamlit run main.py --server.port 8501
```

### Docker Deployment (Optional)
```dockerfile
FROM python:3.9-slim

# Install Poetry
RUN pip install poetry

# Copy project files
COPY . /app
WORKDIR /app

# Install dependencies
RUN poetry config virtualenvs.create false
RUN poetry install --no-dev

# Run application
EXPOSE 8501
CMD ["poetry", "run", "streamlit", "run", "main.py"]
```

## 📝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and test thoroughly
4. Run code formatting: `poetry run black .`
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🆘 Support

If you encounter issues:
1. Check this troubleshooting section
2. Verify your AWS setup and permissions
3. Enable debug mode to inspect context
4. Check the application logs
5. Create an issue with detailed error information

---

**Built with**: Python, Streamlit, LangGraph, AWS Bedrock, SQLite, Poetry