// Chat Application JavaScript
class ChatApp {
    constructor() {
        this.socket = null;
        this.currentChatId = null;
        this.isTyping = false;
        this.debugMode = false;
        
        this.initializeElements();
        this.initializeSocket();
        this.bindEvents();
        this.initializeMarked();
        this.initializeHighlight();
    }
    
    initializeElements() {
        // Chat elements
        this.chatMessages = document.getElementById('chatMessages');
        this.messageInput = document.getElementById('messageInput');
        this.sendBtn = document.getElementById('sendBtn');
        this.typingIndicator = document.getElementById('typingIndicator');
        
        // Sidebar elements
        this.newChatBtn = document.getElementById('newChatBtn');
        this.modelSelect = document.getElementById('modelSelect');
        this.chatTypeSelect = document.getElementById('chatTypeSelect');
        this.recentChats = document.getElementById('recentChats');
        
        // Debug elements
        this.debugMode = document.getElementById('debugMode');
        this.debugPanel = document.getElementById('debugPanel');
        this.contextInfo = document.getElementById('contextInfo');
        this.sessionInfo = document.getElementById('sessionInfo');
        
        // Status elements
        this.statusText = document.getElementById('statusText');
    }
    
    initializeSocket() {
        this.socket = io();
        
        this.socket.on('connect', () => {
            console.log('Connected to server');
            this.updateStatus('Connected to server');
        });
        
        this.socket.on('disconnect', () => {
            console.log('Disconnected from server');
            this.updateStatus('Disconnected from server');
        });
        
        this.socket.on('status', (data) => {
            if (data.credentials_valid) {
                this.updateStatus('Ready to chat');
            } else {
                this.updateStatus('Please configure AWS credentials');
            }
        });
        
        this.socket.on('message', (data) => {
            this.addMessage(data.role, data.content, data.timestamp, data);
            this.scrollToBottom();
        });
        
        this.socket.on('typing', (data) => {
            this.setTypingIndicator(data.typing);
        });
        
        this.socket.on('agent_thinking', (data) => {
            this.setAgentThinking(data.thinking, data.step);
        });
        
        this.socket.on('debug_info', (data) => {
            this.updateDebugInfo(data);
        });
        
        this.socket.on('error', (data) => {
            this.showError(data.message);
            this.setTypingIndicator(false);
            this.setAgentThinking(false);
        });
    }
    
    bindEvents() {
        // Send message events
        this.sendBtn.addEventListener('click', () => this.sendMessage());
        this.messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        
        // New chat button
        if (this.newChatBtn) {
            this.newChatBtn.addEventListener('click', () => this.startNewChat());
        }
        
        // Debug mode toggle
        if (this.debugMode) {
            this.debugMode.addEventListener('change', (e) => {
                this.toggleDebugMode(e.target.checked);
            });
        }
        
        // Chat management buttons
        this.bindChatEvents();
    }
    
    bindChatEvents() {
        // Load chat buttons
        document.querySelectorAll('.chat-load-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const chatId = parseInt(e.currentTarget.dataset.chatId);
                this.loadChat(chatId);
            });
        });
        
        // Delete chat buttons
        document.querySelectorAll('.chat-delete-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const chatId = parseInt(e.currentTarget.dataset.chatId);
                this.deleteChat(chatId);
            });
        });
    }
    
    initializeMarked() {
        if (typeof marked !== 'undefined') {
            marked.setOptions({
                highlight: function(code, lang) {
                    if (typeof hljs !== 'undefined' && lang && hljs.getLanguage(lang)) {
                        try {
                            return hljs.highlight(code, { language: lang }).value;
                        } catch (err) {}
                    }
                    return code;
                },
                breaks: true,
                gfm: true
            });
        }
    }
    
    initializeHighlight() {
        if (typeof hljs !== 'undefined') {
            hljs.highlightAll();
        }
    }
    
    sendMessage() {
        const message = this.messageInput.value.trim();
        if (!message || this.isTyping) return;
        
        const modelId = this.modelSelect ? this.modelSelect.value : null;
        const modelName = this.modelSelect ? this.modelSelect.options[this.modelSelect.selectedIndex].text : null;
        const chatType = this.chatTypeSelect ? this.chatTypeSelect.value : 'General Chat';
        
        if (!modelId) {
            this.showError('Please select a model');
            return;
        }
        
        // Clear input
        this.messageInput.value = '';
        
        // Send message via socket
        this.socket.emit('send_message', {
            message: message,
            model_id: modelId,
            model_name: modelName,
            chat_type: chatType
        });
        
        this.isTyping = true;
        this.updateSendButton(false);
    }
    
    addMessage(role, content, timestamp, data = {}) {
        // Clear welcome message if it exists
        const welcomeMessage = this.chatMessages.querySelector('.text-center');
        if (welcomeMessage) {
            welcomeMessage.remove();
        }
        
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}-message`;
        
        // Add avatar
        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        if (role === 'user') {
            avatar.innerHTML = '<i class="fas fa-user"></i>';
        } else {
            avatar.innerHTML = '<i class="fas fa-robot"></i>';
        }
        
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        
        // Handle tool information for assistant messages
        if (role === 'assistant' && data.has_tools) {
            this.createToolAwareMessage(messageContent, content, data);
        } else {
            // Regular message handling
            if (role === 'assistant') {
                messageContent.innerHTML = marked.parse(content);
                // Highlight code blocks
                messageContent.querySelectorAll('pre code').forEach((block) => {
                    hljs.highlightElement(block);
                });
            } else {
                messageContent.textContent = content;
            }
        }
        
        const messageTimestamp = document.createElement('div');
        messageTimestamp.className = 'message-timestamp';
        messageTimestamp.textContent = this.formatTimestamp(timestamp);
        
        messageDiv.appendChild(avatar);
        messageDiv.appendChild(messageContent);
        messageDiv.appendChild(messageTimestamp);
        
        this.chatMessages.appendChild(messageDiv);
        this.scrollToBottom();
        
        // Reset typing state
        if (role === 'assistant') {
            this.isTyping = false;
            this.updateSendButton(true);
        }
    }
    
    createToolAwareMessage(container, content, data) {
        // Create main response content
        const responseSection = document.createElement('div');
        responseSection.className = 'tool-response-section';
        responseSection.innerHTML = marked.parse(content);
        
        // Highlight code blocks in response
        responseSection.querySelectorAll('pre code').forEach((block) => {
            hljs.highlightElement(block);
        });
        
        // Create tool information section
        if (data.tools_used && data.tools_used.length > 0) {
            const toolSection = document.createElement('div');
            toolSection.className = 'tool-section';
            
            // Tool section header
            const toolHeader = document.createElement('div');
            toolHeader.className = 'tool-header';
            toolHeader.innerHTML = `
                <div class="tool-header-content">
                    <i class="fas fa-tools"></i>
                    <span>Tools Used (${data.tools_used.length})</span>
                    <span class="tool-iterations">${data.iterations || 0} iterations</span>
                </div>
                <button class="btn btn-sm btn-outline-secondary tool-toggle" onclick="this.closest('.tool-section').querySelector('.tool-details').classList.toggle('collapsed')">
                    <i class="fas fa-chevron-down"></i>
                </button>
            `;
            
            // Tool details (collapsible)
            const toolDetails = document.createElement('div');
            toolDetails.className = 'tool-details collapsed';
            
            // Add reasoning section if available
            if (data.reasoning) {
                const reasoningDiv = document.createElement('div');
                reasoningDiv.className = 'tool-reasoning';
                reasoningDiv.innerHTML = `
                    <div class="reasoning-header">
                        <i class="fas fa-brain"></i>
                        <span>Reasoning Process</span>
                    </div>
                    <div class="reasoning-content">${this.formatReasoning(data.reasoning)}</div>
                `;
                toolDetails.appendChild(reasoningDiv);
            }
            
            // Add tool results
            if (data.tool_results && data.tool_results.length > 0) {
                data.tool_results.forEach((result, index) => {
                    const toolResult = document.createElement('div');
                    toolResult.className = 'tool-result';
                    
                    const toolIcon = this.getToolIcon(result.tool);
                    const toolName = this.getToolDisplayName(result.tool);
                    
                    toolResult.innerHTML = `
                        <div class="tool-result-header">
                            <div class="tool-info">
                                <i class="${toolIcon}"></i>
                                <span class="tool-name">${toolName}</span>
                                <span class="tool-step">Step ${index + 1}</span>
                            </div>
                            <button class="btn btn-sm btn-outline-secondary result-toggle" onclick="this.closest('.tool-result').querySelector('.tool-result-details').classList.toggle('collapsed')">
                                <i class="fas fa-chevron-down"></i>
                            </button>
                        </div>
                        <div class="tool-result-details collapsed">
                            <div class="tool-input">
                                <strong>Input:</strong>
                                <div class="tool-input-content">${this.escapeHtml(result.input)}</div>
                            </div>
                            <div class="tool-output">
                                <strong>Output:</strong>
                                <div class="tool-output-content">${this.formatToolOutput(result.output)}</div>
                            </div>
                        </div>
                    `;
                    
                    toolDetails.appendChild(toolResult);
                });
            }
            
            toolSection.appendChild(toolHeader);
            toolSection.appendChild(toolDetails);
            
            container.appendChild(toolSection);
        }
        
        container.appendChild(responseSection);
    }
    
    formatReasoning(reasoning) {
        // Format reasoning text with basic markdown support
        return reasoning
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/\n/g, '<br>');
    }
    
    formatToolOutput(output) {
        // Format tool output with proper escaping and basic formatting
        const escaped = this.escapeHtml(output);
        return escaped.replace(/\n/g, '<br>');
    }
    
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
    
    getToolIcon(toolName) {
        const icons = {
            'web_search': 'fas fa-search',
            'calculator': 'fas fa-calculator',
            'code_executor': 'fas fa-code',
            'weather_api': 'fas fa-cloud-sun',
            'file_reader': 'fas fa-file-alt',
            'database_query': 'fas fa-database',
            'none': 'fas fa-cog'
        };
        return icons[toolName] || 'fas fa-tool';
    }
    
    getToolDisplayName(toolName) {
        const names = {
            'web_search': 'Web Search',
            'calculator': 'Calculator',
            'code_executor': 'Code Executor',
            'weather_api': 'Weather API',
            'file_reader': 'File Reader',
            'database_query': 'Database Query',
            'none': 'No Tool'
        };
        return names[toolName] || toolName;
    }
    
    setTypingIndicator(show) {
        if (this.typingIndicator) {
            this.typingIndicator.style.display = show ? 'flex' : 'none';
            if (show) {
                this.scrollToBottom();
            }
        }
    }
    
    setAgentThinking(thinking, step = '') {
        let thinkingIndicator = document.getElementById('agentThinking');
        
        if (thinking) {
            if (!thinkingIndicator) {
                thinkingIndicator = document.createElement('div');
                thinkingIndicator.id = 'agentThinking';
                thinkingIndicator.className = 'agent-thinking';
                thinkingIndicator.innerHTML = `
                    <div class="thinking-content">
                        <div class="thinking-spinner">
                            <div class="spinner-border spinner-border-sm" role="status"></div>
                        </div>
                        <div class="thinking-text">
                            <span class="thinking-step">Thinking...</span>
                            <span class="thinking-detail">Analyzing your request</span>
                        </div>
                    </div>
                `;
                this.chatMessages.appendChild(thinkingIndicator);
            }
            
            // Update thinking step
            const stepElement = thinkingIndicator.querySelector('.thinking-step');
            const detailElement = thinkingIndicator.querySelector('.thinking-detail');
            
            switch(step) {
                case 'reasoning':
                    stepElement.textContent = 'Reasoning...';
                    detailElement.textContent = 'Analyzing your request';
                    break;
                case 'planning':
                    stepElement.textContent = 'Planning...';
                    detailElement.textContent = 'Deciding which tools to use';
                    break;
                case 'executing':
                    stepElement.textContent = 'Executing...';
                    detailElement.textContent = 'Running tools and gathering information';
                    break;
                case 'observing':
                    stepElement.textContent = 'Observing...';
                    detailElement.textContent = 'Analyzing tool results';
                    break;
                default:
                    stepElement.textContent = 'Thinking...';
                    detailElement.textContent = 'Processing your request';
            }
            
            this.scrollToBottom();
        } else {
            if (thinkingIndicator) {
                thinkingIndicator.remove();
            }
        }
    }
    
    updateSendButton(enabled) {
        if (this.sendBtn) {
            this.sendBtn.disabled = !enabled;
            if (enabled) {
                this.sendBtn.innerHTML = '<i class="fas fa-paper-plane"></i>';
            } else {
                this.sendBtn.innerHTML = '<div class="spinner-border spinner-border-sm" role="status"></div>';
            }
        }
    }
    
    updateStatus(message) {
        if (this.statusText) {
            this.statusText.textContent = message;
        }
    }
    
    showError(message) {
        // Create error message
        const errorDiv = document.createElement('div');
        errorDiv.className = 'alert alert-danger alert-dismissible fade show';
        errorDiv.innerHTML = `
            <i class="fas fa-exclamation-circle"></i> ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        // Insert at top of chat messages
        this.chatMessages.insertBefore(errorDiv, this.chatMessages.firstChild);
        
        // Auto-dismiss after 5 seconds
        setTimeout(() => {
            if (errorDiv.parentNode) {
                errorDiv.remove();
            }
        }, 5000);
    }
    
    scrollToBottom() {
        setTimeout(() => {
            this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
        }, 100);
    }
    
    formatTimestamp(timestamp) {
        const date = new Date(timestamp);
        return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }
    
    toggleDebugMode(enabled) {
        if (this.debugPanel) {
            this.debugPanel.style.display = enabled ? 'block' : 'none';
        }
    }
    
    updateDebugInfo(data) {
        if (this.contextInfo) {
            this.contextInfo.textContent = JSON.stringify(data, null, 2);
        }
        
        if (this.sessionInfo) {
            const sessionData = {
                current_chat_id: this.currentChatId,
                is_typing: this.isTyping,
                debug_mode: this.debugMode ? this.debugMode.checked : false,
                timestamp: new Date().toISOString()
            };
            this.sessionInfo.textContent = JSON.stringify(sessionData, null, 2);
        }
    }
    
    async startNewChat() {
        try {
            const response = await fetch('/new_chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            });
            
            if (response.ok) {
                this.currentChatId = null;
                this.clearChatMessages();
                this.updateActiveChatButton(null);
                this.updateStatus('New chat started');
                
                // Refresh chat list
                await this.refreshChatList();
            } else {
                throw new Error('Failed to start new chat');
            }
        } catch (error) {
            console.error('Error starting new chat:', error);
            this.showError('Failed to start new chat');
        }
    }
    
    async refreshChatList() {
        try {
            const response = await fetch('/api/chats');
            const data = await response.json();
            
            if (data.success) {
                this.updateChatList(data.chats);
                this.updateChatCount(data.chats.length);
            }
        } catch (error) {
            console.error('Error refreshing chat list:', error);
        }
    }
    
    updateChatList(chats) {
        const chatList = document.getElementById('recentChats');
        if (!chatList) return;
        
        if (chats.length === 0) {
            chatList.innerHTML = `
                <div class="empty-state">
                    <i class="fas fa-comments"></i>
                    <p>No chats yet</p>
                    <small>Start a new conversation!</small>
                </div>
            `;
            return;
        }
        
        chatList.innerHTML = chats.map(chat => `
            <div class="chat-item" data-chat-id="${chat.id}">
                <div class="chat-item-content">
                    <button class="chat-load-btn" 
                            data-chat-id="${chat.id}" 
                            title="Created: ${chat.created_at}&#10;Messages: ${chat.message_count}">
                        <div class="chat-icon">
                            <i class="fas fa-comment"></i>
                        </div>
                        <div class="chat-info">
                            <div class="chat-name">${chat.name.length > 30 ? chat.name.substring(0, 30) + '...' : chat.name}</div>
                            <div class="chat-meta">${chat.message_count} messages</div>
                        </div>
                    </button>
                    <div class="chat-actions">
                        <button class="btn btn-sm btn-outline-danger chat-delete-btn" 
                                data-chat-id="${chat.id}"
                                title="Delete chat">
                            <i class="fas fa-trash"></i>
                        </button>
                    </div>
                </div>
            </div>
        `).join('');
        
        // Re-bind event listeners
        this.bindChatListEvents();
    }
    
    updateChatCount(count) {
        const chatCount = document.getElementById('chatCount');
        if (chatCount) {
            chatCount.textContent = count;
        }
    }
    
    bindChatListEvents() {
        // Chat load buttons
        document.querySelectorAll('.chat-load-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const chatId = parseInt(e.currentTarget.dataset.chatId);
                this.loadChat(chatId);
            });
        });
        
        // Chat delete buttons
        document.querySelectorAll('.chat-delete-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                const chatId = parseInt(e.currentTarget.dataset.chatId);
                this.deleteChat(chatId);
            });
        });
    }
    
    async loadChat(chatId) {
        try {
            const response = await fetch(`/chat/${chatId}`);
            const data = await response.json();
            
            if (data.success) {
                this.currentChatId = chatId;
                this.clearChatMessages();
                
                // Load messages with tool information
                data.messages.forEach(msg => {
                    this.addMessage(msg.role, msg.content, msg.timestamp, msg);
                });
                
                this.updateActiveChatButton(chatId);
                this.updateStatus(`Loaded chat ${chatId}`);
            } else {
                throw new Error(data.error || 'Failed to load chat');
            }
        } catch (error) {
            console.error('Error loading chat:', error);
            this.showError('Failed to load chat');
        }
    }
    
    async deleteChat(chatId) {
        if (!confirm('Are you sure you want to delete this chat?')) {
            return;
        }
        
        try {
            const response = await fetch(`/chat/${chatId}/delete`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            });
            
            if (response.ok) {
                // Remove chat from sidebar
                const chatItem = document.querySelector(`[data-chat-id="${chatId}"]`);
                if (chatItem) {
                    chatItem.remove();
                }
                
                // If this was the current chat, start a new one
                if (this.currentChatId === chatId) {
                    this.currentChatId = null;
                    this.clearChatMessages();
                }
                
                this.updateStatus('Chat deleted');
            } else {
                throw new Error('Failed to delete chat');
            }
        } catch (error) {
            console.error('Error deleting chat:', error);
            this.showError('Failed to delete chat');
        }
    }
    
    clearChatMessages() {
        if (this.chatMessages) {
            this.chatMessages.innerHTML = `
                <div class="welcome-message">
                    <div class="welcome-content">
                        <div class="welcome-icon">
                            <i class="fas fa-robot"></i>
                        </div>
                        <h3>Welcome to Bedrock Chat Assistant!</h3>
                        <p>I'm an advanced AI assistant powered by AWS Bedrock with tool-calling capabilities.</p>
                        <div class="welcome-features">
                            <div class="feature-item">
                                <i class="fas fa-search"></i>
                                <span>Web Search</span>
                            </div>
                            <div class="feature-item">
                                <i class="fas fa-calculator"></i>
                                <span>Calculations</span>
                            </div>
                            <div class="feature-item">
                                <i class="fas fa-code"></i>
                                <span>Code Execution</span>
                            </div>
                            <div class="feature-item">
                                <i class="fas fa-cloud-sun"></i>
                                <span>Weather Data</span>
                            </div>
                        </div>
                        <p class="mt-3">
                            <strong>Ask me anything!</strong> I can help with research, calculations, coding, and more.
                        </p>
                    </div>
                </div>
            `;
        }
    }
    
    updateActiveChatButton(chatId) {
        // Remove active class from all chat buttons
        document.querySelectorAll('.chat-load-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        
        // Add active class to current chat button
        if (chatId) {
            const activeBtn = document.querySelector(`[data-chat-id="${chatId}"].chat-load-btn`);
            if (activeBtn) {
                activeBtn.classList.add('active');
            }
        }
    }
    
    exportChat() {
        const messages = this.chatMessages.querySelectorAll('.message');
        if (messages.length === 0) {
            this.showError('No messages to export');
            return;
        }
        
        let chatContent = 'Bedrock Chat Export\n';
        chatContent += '===================\n\n';
        
        messages.forEach(msg => {
            const role = msg.classList.contains('user-message') ? 'User' : 'Assistant';
            const content = msg.querySelector('.message-content').textContent;
            const timestamp = msg.querySelector('.message-timestamp').textContent;
            
            chatContent += `${role} (${timestamp}):\n${content}\n\n`;
        });
        
        const blob = new Blob([chatContent], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `bedrock-chat-${new Date().toISOString().split('T')[0]}.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new ChatApp();
});

// Handle responsive sidebar toggle
document.addEventListener('DOMContentLoaded', () => {
    const sidebar = document.querySelector('.sidebar');
    const mainContent = document.querySelector('.main-content');
    
    // Add mobile menu button if on mobile
    if (window.innerWidth <= 768) {
        const chatHeader = document.querySelector('.chat-header');
        if (chatHeader && !chatHeader.querySelector('.mobile-menu-btn')) {
            const menuBtn = document.createElement('button');
            menuBtn.className = 'mobile-menu-btn';
            menuBtn.innerHTML = '<i class="fas fa-bars"></i>';
            menuBtn.addEventListener('click', () => {
                sidebar.classList.toggle('show');
            });
            chatHeader.appendChild(menuBtn);
        }
    }
    
    // Handle window resize
    window.addEventListener('resize', () => {
        if (window.innerWidth > 768) {
            sidebar.classList.remove('show');
        }
    });
    
    // Close sidebar when clicking outside on mobile
    document.addEventListener('click', (e) => {
        if (window.innerWidth <= 768 && 
            sidebar.classList.contains('show') && 
            !sidebar.contains(e.target) && 
            !e.target.closest('.mobile-menu-btn')) {
            sidebar.classList.remove('show');
        }
    });
}); 