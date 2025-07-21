"""
SQLite Database for Chat Persistence
Handles all database operations for chats and messages
"""

import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from contextlib import contextmanager

from config import Config

# Configure logging
logger = logging.getLogger(__name__)

class DatabaseError(Exception):
    """Custom exception for database operations"""
    pass

class ChatDatabase:
    """
    SQLite database manager for chat persistence
    Provides all CRUD operations for chats and messages
    """
    
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = Path(db_path) if db_path else Config.get_database_path()
        self.init_database()
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = None
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row  # Enable column access by name
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            raise DatabaseError(f"Database operation failed: {str(e)}")
        finally:
            if conn:
                conn.close()
    
    def init_database(self) -> None:
        """Initialize database with required tables and indexes"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Create chats table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS chats (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        chat_type TEXT DEFAULT 'General Chat',
                        model_name TEXT DEFAULT '',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        message_count INTEGER DEFAULT 0,
                        is_active BOOLEAN DEFAULT 1
                    )
                ''')
                
                # Create messages table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        chat_id INTEGER NOT NULL,
                        role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
                        content TEXT NOT NULL,
                        metadata TEXT DEFAULT NULL,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        token_count INTEGER DEFAULT 0,
                        FOREIGN KEY (chat_id) REFERENCES chats (id) ON DELETE CASCADE
                    )
                ''')
                
                # Add metadata column if it doesn't exist (for existing databases)
                try:
                    cursor.execute('ALTER TABLE messages ADD COLUMN metadata TEXT DEFAULT NULL')
                except:
                    pass  # Column already exists
                
                # Create indexes for performance
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_messages_chat_id 
                    ON messages (chat_id)
                ''')
                
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_messages_timestamp 
                    ON messages (timestamp DESC)
                ''')
                
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_chats_updated_at 
                    ON chats (updated_at DESC)
                ''')
                
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_chats_active 
                    ON chats (is_active, updated_at DESC)
                ''')
                
                conn.commit()
                logger.info(f"Database initialized successfully at {self.db_path}")
                
        except Exception as e:
            logger.error(f"Failed to initialize database: {str(e)}")
            raise DatabaseError(f"Database initialization failed: {str(e)}")
    
    def create_chat(
        self, 
        first_message: str, 
        chat_type: str = "General Chat",
        model_name: str = ""
    ) -> int:
        """
        Create a new chat
        
        Args:
            first_message: First message to generate chat name from
            chat_type: Type of chat (from Config.CHAT_TYPES)
            model_name: Name of the AI model being used
            
        Returns:
            ID of the newly created chat
            
        Raises:
            DatabaseError: If chat creation fails
        """
        try:
            chat_name = self._generate_chat_name(first_message)
            
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO chats (name, chat_type, model_name, message_count) 
                    VALUES (?, ?, ?, 0)
                ''', (chat_name, chat_type, model_name))
                
                chat_id = cursor.lastrowid
                conn.commit()
                
                logger.info(f"Created new chat with ID: {chat_id}, name: '{chat_name}'")
                return chat_id
                
        except Exception as e:
            logger.error(f"Failed to create chat: {str(e)}")
            raise DatabaseError(f"Chat creation failed: {str(e)}")
    
    def add_message(
        self, 
        chat_id: int, 
        role: str, 
        content: str,
        token_count: int = 0,
        metadata: str = None
    ) -> int:
        """
        Add a message to a chat
        
        Args:
            chat_id: ID of the chat
            role: Message role ('user', 'assistant', 'system')
            content: Message content
            token_count: Estimated token count for the message
            metadata: JSON string containing thinking/tool parts
            
        Returns:
            ID of the newly created message
            
        Raises:
            DatabaseError: If message addition fails
        """
        try:
            if role not in ['user', 'assistant', 'system']:
                raise ValueError(f"Invalid role: {role}")
            
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Add message
                cursor.execute('''
                    INSERT INTO messages (chat_id, role, content, token_count, metadata) 
                    VALUES (?, ?, ?, ?, ?)
                ''', (chat_id, role, content, token_count, metadata))
                
                message_id = cursor.lastrowid
                
                # Update chat statistics
                cursor.execute('''
                    UPDATE chats 
                    SET message_count = message_count + 1, 
                        updated_at = CURRENT_TIMESTAMP 
                    WHERE id = ?
                ''', (chat_id,))
                
                conn.commit()
                logger.debug(f"Added {role} message to chat {chat_id}")
                return message_id
                
        except Exception as e:
            logger.error(f"Failed to add message: {str(e)}")
            raise DatabaseError(f"Message addition failed: {str(e)}")
    
    def get_chat_messages(self, chat_id: int) -> List[Dict]:
        """
        Get all messages for a chat in chronological order
        
        Args:
            chat_id: ID of the chat
            
        Returns:
            List of message dictionaries
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT id, role, content, timestamp, token_count, metadata 
                    FROM messages 
                    WHERE chat_id = ? 
                    ORDER BY timestamp ASC
                ''', (chat_id,))
                
                messages = []
                for row in cursor.fetchall():
                    messages.append({
                        'id': row['id'],
                        'role': row['role'],
                        'content': row['content'],
                        'timestamp': row['timestamp'],
                        'token_count': row['token_count'],
                        'metadata': row['metadata']
                    })
                
                logger.debug(f"Retrieved {len(messages)} messages for chat {chat_id}")
                return messages
                
        except Exception as e:
            logger.error(f"Failed to get chat messages: {str(e)}")
            return []
    
    def get_recent_messages(self, chat_id: int, count: int = 5) -> List[Dict]:
        """
        Get the most recent messages for a chat
        
        Args:
            chat_id: ID of the chat
            count: Number of recent messages to retrieve
            
        Returns:
            List of recent message dictionaries in chronological order
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT id, role, content, timestamp, token_count 
                    FROM messages 
                    WHERE chat_id = ? 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (chat_id, count))
                
                messages = []
                for row in cursor.fetchall():
                    messages.append({
                        'id': row['id'],
                        'role': row['role'],
                        'content': row['content'],
                        'timestamp': row['timestamp'],
                        'token_count': row['token_count']
                    })
                
                # Reverse to get chronological order
                messages.reverse()
                
                logger.debug(f"Retrieved {len(messages)} recent messages for chat {chat_id}")
                return messages
                
        except Exception as e:
            logger.error(f"Failed to get recent messages: {str(e)}")
            return []
    
    def get_recent_chats(self, limit: int = 10) -> List[Dict]:
        """
        Get recently updated chats
        
        Args:
            limit: Maximum number of chats to return
            
        Returns:
            List of chat dictionaries ordered by most recent update
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT id, name, chat_type, model_name, created_at, 
                           updated_at, message_count, is_active 
                    FROM chats 
                    WHERE is_active = 1 
                    ORDER BY updated_at DESC 
                    LIMIT ?
                ''', (limit,))
                
                chats = []
                for row in cursor.fetchall():
                    chats.append({
                        'id': row['id'],
                        'name': row['name'],
                        'chat_type': row['chat_type'],
                        'model_name': row['model_name'],
                        'created_at': row['created_at'],
                        'updated_at': row['updated_at'],
                        'message_count': row['message_count'],
                        'is_active': bool(row['is_active'])
                    })
                
                logger.debug(f"Retrieved {len(chats)} recent chats")
                return chats
                
        except Exception as e:
            logger.error(f"Failed to get recent chats: {str(e)}")
            return []
    
    def get_chat_list(self, limit: Optional[int] = None, active_only: bool = True) -> List[Dict]:
        """
        Get list of chats with optional filtering
        
        Args:
            limit: Maximum number of chats to return
            active_only: Only return active chats
            
        Returns:
            List of chat dictionaries
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                query = '''
                    SELECT id, name, chat_type, model_name, created_at, 
                           updated_at, message_count, is_active 
                    FROM chats
                '''
                params = []
                
                if active_only:
                    query += ' WHERE is_active = 1'
                
                query += ' ORDER BY updated_at DESC'
                
                if limit:
                    query += ' LIMIT ?'
                    params.append(limit)
                
                cursor.execute(query, params)
                
                chats = []
                for row in cursor.fetchall():
                    chats.append({
                        'id': row['id'],
                        'name': row['name'],
                        'chat_type': row['chat_type'],
                        'model_name': row['model_name'],
                        'created_at': row['created_at'],
                        'updated_at': row['updated_at'],
                        'message_count': row['message_count'],
                        'is_active': bool(row['is_active'])
                    })
                
                logger.debug(f"Retrieved {len(chats)} chats")
                return chats
                
        except Exception as e:
            logger.error(f"Failed to get chat list: {str(e)}")
            return []
    
    def get_chat_info(self, chat_id: int) -> Optional[Dict]:
        """
        Get detailed information about a specific chat
        
        Args:
            chat_id: ID of the chat
            
        Returns:
            Chat information dictionary or None if not found
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT id, name, chat_type, model_name, created_at, 
                           updated_at, message_count, is_active 
                    FROM chats 
                    WHERE id = ?
                ''', (chat_id,))
                
                row = cursor.fetchone()
                if row:
                    return {
                        'id': row['id'],
                        'name': row['name'],
                        'chat_type': row['chat_type'],
                        'model_name': row['model_name'],
                        'created_at': row['created_at'],
                        'updated_at': row['updated_at'],
                        'message_count': row['message_count'],
                        'is_active': bool(row['is_active'])
                    }
                
                return None
                
        except Exception as e:
            logger.error(f"Failed to get chat info: {str(e)}")
            return None
    
    def delete_chat(self, chat_id: int) -> bool:
        """
        Delete a chat and all its messages
        
        Args:
            chat_id: ID of the chat to delete
            
        Returns:
            True if deletion was successful, False otherwise
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Delete messages (CASCADE should handle this, but being explicit)
                cursor.execute('DELETE FROM messages WHERE chat_id = ?', (chat_id,))
                
                # Delete chat
                cursor.execute('DELETE FROM chats WHERE id = ?', (chat_id,))
                
                deleted_rows = cursor.rowcount
                conn.commit()
                
                if deleted_rows > 0:
                    logger.info(f"Deleted chat {chat_id}")
                    return True
                else:
                    logger.warning(f"No chat found with ID {chat_id}")
                    return False
                
        except Exception as e:
            logger.error(f"Failed to delete chat: {str(e)}")
            return False
    
    def search_chats(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Search for chats by name or content
        
        Args:
            query: Search query string
            limit: Maximum number of results to return
            
        Returns:
            List of matching chat dictionaries
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Search in chat names and message content
                cursor.execute('''
                    SELECT DISTINCT c.id, c.name, c.chat_type, c.model_name, 
                           c.created_at, c.updated_at, c.message_count, c.is_active
                    FROM chats c
                    LEFT JOIN messages m ON c.id = m.chat_id
                    WHERE c.is_active = 1 AND (
                        c.name LIKE ? OR 
                        m.content LIKE ?
                    )
                    ORDER BY c.updated_at DESC
                    LIMIT ?
                ''', (f'%{query}%', f'%{query}%', limit))
                
                chats = []
                for row in cursor.fetchall():
                    chats.append({
                        'id': row['id'],
                        'name': row['name'],
                        'chat_type': row['chat_type'],
                        'model_name': row['model_name'],
                        'created_at': row['created_at'],
                        'updated_at': row['updated_at'],
                        'message_count': row['message_count'],
                        'is_active': bool(row['is_active'])
                    })
                
                logger.debug(f"Found {len(chats)} chats matching '{query}'")
                return chats
                
        except Exception as e:
            logger.error(f"Failed to search chats: {str(e)}")
            return []
    
    def get_chat_statistics(self) -> Dict:
        """
        Get overall chat statistics
        
        Returns:
            Dictionary containing various statistics
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                stats = {}
                
                # Total chats
                cursor.execute('SELECT COUNT(*) as count FROM chats WHERE is_active = 1')
                stats['total_chats'] = cursor.fetchone()['count']
                
                # Total messages
                cursor.execute('''
                    SELECT COUNT(*) as count FROM messages m
                    JOIN chats c ON m.chat_id = c.id
                    WHERE c.is_active = 1
                ''')
                stats['total_messages'] = cursor.fetchone()['count']
                
                # Average messages per chat
                if stats['total_chats'] > 0:
                    stats['avg_messages_per_chat'] = stats['total_messages'] / stats['total_chats']
                else:
                    stats['avg_messages_per_chat'] = 0
                
                # Chat type distribution
                cursor.execute('''
                    SELECT chat_type, COUNT(*) as count 
                    FROM chats 
                    WHERE is_active = 1 
                    GROUP BY chat_type
                ''')
                stats['chat_type_distribution'] = {
                    row['chat_type']: row['count'] 
                    for row in cursor.fetchall()
                }
                
                # Model usage statistics
                cursor.execute('''
                    SELECT model_name, COUNT(*) as count 
                    FROM chats 
                    WHERE is_active = 1 AND model_name != ''
                    GROUP BY model_name
                ''')
                stats['model_usage'] = {
                    row['model_name']: row['count'] 
                    for row in cursor.fetchall()
                }
                
                # Recent activity (last 7 days)
                cursor.execute('''
                    SELECT COUNT(*) as count 
                    FROM chats 
                    WHERE is_active = 1 
                    AND updated_at >= datetime('now', '-7 days')
                ''')
                stats['recent_chats'] = cursor.fetchone()['count']
                
                # Database file size
                try:
                    stats['database_size_bytes'] = self.db_path.stat().st_size
                except:
                    stats['database_size_bytes'] = 0
                
                stats['database_path'] = str(self.db_path)
                
                logger.debug("Retrieved chat statistics")
                return stats
                
        except Exception as e:
            logger.error(f"Failed to get chat statistics: {str(e)}")
            return {
                'total_chats': 0,
                'total_messages': 0,
                'avg_messages_per_chat': 0,
                'chat_type_distribution': {},
                'model_usage': {},
                'recent_chats': 0,
                'database_size_bytes': 0,
                'database_path': str(self.db_path)
            }
    
    def update_chat_title(self, chat_id: int, title: str) -> bool:
        """
        Update chat title
        
        Args:
            chat_id: ID of the chat
            title: New title for the chat
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE chats 
                    SET name = ?, updated_at = CURRENT_TIMESTAMP 
                    WHERE id = ?
                ''', (title, chat_id))
                
                success = cursor.rowcount > 0
                if success:
                    logger.debug(f"Updated chat {chat_id} title to: {title}")
                return success
                
        except Exception as e:
            logger.error(f"Failed to update chat title: {str(e)}")
            return False
    
    def get_chat_message_count(self, chat_id: int) -> int:
        """
        Get the number of messages in a chat
        
        Args:
            chat_id: ID of the chat
            
        Returns:
            Number of messages in the chat
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT COUNT(*) FROM messages WHERE chat_id = ?', (chat_id,))
                count = cursor.fetchone()[0]
                return count
        except Exception as e:
            logger.error(f"Failed to get message count for chat {chat_id}: {str(e)}")
            return 0
    
    def generate_smart_chat_title(self, user_message: str) -> str:
        """
        Generate a smart chat title from user message like modern LLM apps
        
        Args:
            user_message: The first user message
            
        Returns:
            Generated smart title
        """
        # Clean the message
        message = user_message.strip()
        
        if not message:
            return "New Chat"
        
        # Remove common question words and make it more title-like
        # Common patterns in user messages (improved patterns)
        title_patterns = [
            # Questions - more precise patterns
            (r'^(what is|what are|what\'s)\s+', ''),
            (r'^(how do i|how can i|how to)\s+', ''),
            (r'^(why is|why are|why does|why do)\s+', ''),
            (r'^(when is|when are|when does|when do)\s+', ''),
            (r'^(where is|where are|where can|where do)\s+', ''),
            (r'^(who is|who are)\s+', ''),
            (r'^(which is|which are)\s+', ''),
            
            # Commands and requests
            (r'^(can you|could you|would you|will you)\s+(please\s+)?', ''),
            (r'^(please\s+)?(help me|show me|tell me|explain to me)\s+', ''),
            (r'^(please\s+)?(write|create|generate|make|build|develop)\s+(a|an|me\s+)?', ''),
            (r'^(please\s+)?(explain|describe|summarize)\s+', ''),
            
            # Personal requests
            (r'^(i need|i want|i would like|i\'d like)\s+(to\s+|you to\s+)?', ''),
            (r'^(let\'s|lets)\s+', ''),
            
            # Cleanup
            (r'\?+$', ''),  # Remove trailing question marks
            (r'\s+(please|for me|thanks|thank you)$', ''),  # Remove trailing politeness
            (r'^\s*', ''),  # Remove leading whitespace
        ]
        
        import re
        clean_message = message
        
        # Apply cleaning patterns in order
        for pattern, replacement in title_patterns:
            clean_message = re.sub(pattern, replacement, clean_message, flags=re.IGNORECASE)
        
        clean_message = clean_message.strip()
        
        # If we removed too much, fall back to original
        if len(clean_message) < 2:
            clean_message = message
        
        # Capitalize first letter
        if clean_message:
            clean_message = clean_message[0].upper() + clean_message[1:]
        
        # Take first 5-6 words for a good title length
        words = clean_message.split()
        if len(words) <= 6:
            title = " ".join(words)
        else:
            title = " ".join(words[:6])
            # Add ellipsis if message was much longer
            if len(words) > 8:
                title += "..."
        
        # Ensure reasonable length
        if len(title) > 60:
            title = title[:57] + "..."
        elif len(title) < 3:
            # Fallback to original method
            title = self._generate_chat_name(message)
        
        return title
    
    def _generate_chat_name(self, first_message: str) -> str:
        """
        Generate a chat name from the first message
        
        Args:
            first_message: Content of the first message
            
        Returns:
            Generated chat name
        """
        # Clean and split into words
        words = first_message.strip().split()[:4]  # Take first 4 words
        
        if not words:
            return "New Chat"
        
        chat_name = " ".join(words)
        
        # Add ellipsis if message was longer
        if len(first_message.strip().split()) > 4:
            chat_name += "..."
        
        # Ensure reasonable length constraints
        if len(chat_name) < 3:
            chat_name = "New Chat"
        elif len(chat_name) > 50:
            chat_name = chat_name[:47] + "..."
        
        return chat_name