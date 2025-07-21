#!/usr/bin/env python3
"""
Test script for real-time thinking streaming
Tests the new character-by-character thinking stream
"""

import asyncio
import time
from bedrock_client import BedrockClient
from chat_agent import ChatAgent
from langchain_core.messages import HumanMessage

async def test_real_time_thinking_streaming():
    """Test the new real-time thinking streaming functionality"""
    print("🧪 Testing Real-Time Thinking Streaming")
    print("=" * 60)
    
    client = BedrockClient()
    agent = ChatAgent(client)
    
    # Prepare test state
    state = {
        'messages': [HumanMessage(content="Explain how machine learning works and calculate 25 * 67")],
        'chat_type': 'Tool-Assisted Research',
        'model_id': 'anthropic.claude-3-haiku-20240307-v1:0',
        'total_messages': 1,
        'context_messages': 0,
        'estimated_tokens': 0,
        'cache_used': False,
        'iteration_count': 0,
        'max_iterations': 3,
        'tools_used': [],
        'tool_results': [],
        'current_thought': '',
        'next_action': '',
        'action_input': '',
        'observation': '',
        'final_answer': ''
    }
    
    print("🧠 Testing reasoning streaming...")
    print("-" * 40)
    
    reasoning_text = ""
    chunk_count = 0
    start_time = time.time()
    
    try:
        async for chunk in agent._stream_reasoning_node(state):
            chunk_count += 1
            reasoning_text += chunk
            current_time = time.time()
            
            # Show each chunk with timing
            print(f"[{current_time - start_time:.3f}s] Chunk {chunk_count}: '{chunk}' ({len(chunk)} chars)")
            
            # Small delay to see streaming effect
            await asyncio.sleep(0.01)
        
        end_time = time.time()
        print("-" * 40)
        print(f"✅ Reasoning completed in {end_time - start_time:.2f}s")
        print(f"📊 Total chunks: {chunk_count}")
        print(f"📝 Total reasoning text: {len(reasoning_text)} characters")
        print(f"🎯 First 200 chars: {reasoning_text[:200]}...")
        
        # Test action planning if reasoning suggests tool use
        if 'tool' in reasoning_text.lower() or 'calculator' in reasoning_text.lower():
            print("\n🔧 Testing action planning streaming...")
            print("-" * 40)
            
            planning_text = ""
            chunk_count = 0
            start_time = time.time()
            
            async for chunk in agent._stream_action_planning_node(state):
                chunk_count += 1
                planning_text += chunk
                current_time = time.time()
                
                print(f"[{current_time - start_time:.3f}s] Planning Chunk {chunk_count}: '{chunk}' ({len(chunk)} chars)")
                await asyncio.sleep(0.01)
            
            end_time = time.time()
            print("-" * 40)
            print(f"✅ Action planning completed in {end_time - start_time:.2f}s")
            print(f"📊 Total planning chunks: {chunk_count}")
            print(f"📝 Planning text: {planning_text}")
        
        # Test response streaming
        print("\n💬 Testing response streaming...")
        print("-" * 40)
        
        response_text = ""
        chunk_count = 0
        start_time = time.time()
        
        async for chunk in agent._stream_response_node(state):
            chunk_count += 1
            response_text += chunk
            current_time = time.time()
            
            print(f"[{current_time - start_time:.3f}s] Response Chunk {chunk_count}: '{chunk}' ({len(chunk)} chars)")
            await asyncio.sleep(0.01)
        
        end_time = time.time()
        print("-" * 40)
        print(f"✅ Response completed in {end_time - start_time:.2f}s")
        print(f"📊 Total response chunks: {chunk_count}")
        print(f"📝 First 200 chars: {response_text[:200]}...")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")

def main():
    """Run the real-time thinking test"""
    print("🚀 Real-Time Thinking Stream Test")
    print("=" * 80)
    
    asyncio.run(test_real_time_thinking_streaming())
    
    print("\n" + "=" * 80)
    print("📋 REAL-TIME THINKING ANALYSIS:")
    print("✅ What we're now testing:")
    print("   - Character-by-character reasoning streaming")
    print("   - Real-time action planning streaming")  
    print("   - Token-by-token response streaming")
    print("   - All phases stream independently")
    
    print("\n🎯 Expected behavior:")
    print("   - Reasoning appears as agent thinks")
    print("   - Planning shows tool selection in real-time")
    print("   - Response streams like human typing")
    print("   - No more 'instant' thought emission")

if __name__ == "__main__":
    main() 