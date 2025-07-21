#!/usr/bin/env python3
"""
Test script to analyze real-time streaming behavior
Tests each component of the streaming pipeline
"""

import asyncio
import time
import json
from datetime import datetime
from bedrock_client import BedrockClient
from chat_agent import ChatAgent
from langchain_core.messages import HumanMessage, SystemMessage

def test_bedrock_streaming():
    """Test raw Bedrock streaming to see if it's truly real-time"""
    print("🧪 Testing Raw Bedrock Streaming...")
    
    client = BedrockClient()
    
    # Test with a longer prompt to see streaming behavior
    test_messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="Explain photosynthesis in detail, step by step, with examples. Make it comprehensive.")
    ]
    
    model_id = "anthropic.claude-3-haiku-20240307-v1:0"
    
    print(f"Testing with model: {model_id}")
    print("=" * 60)
    
    start_time = time.time()
    total_chunks = 0
    first_chunk_time = None
    chunk_times = []
    
    try:
        for chunk in client.invoke_model_stream(model_id, test_messages, max_tokens=1000):
            current_time = time.time()
            
            if first_chunk_time is None:
                first_chunk_time = current_time
                print(f"⏱️  Time to first chunk: {first_chunk_time - start_time:.2f}s")
            
            chunk_times.append(current_time)
            total_chunks += 1
            
            # Show chunk with timing
            print(f"[{current_time - start_time:.2f}s] Chunk {total_chunks}: '{chunk}' ({len(chunk)} chars)")
            
            # Add small delay to see if chunks come independently
            time.sleep(0.01)
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print("=" * 60)
    print(f"📊 Streaming Analysis:")
    print(f"   Total chunks: {total_chunks}")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Time to first chunk: {first_chunk_time - start_time:.2f}s")
    print(f"   Average chunk interval: {total_time / max(total_chunks, 1):.3f}s")
    
    # Analyze chunk intervals
    if len(chunk_times) > 1:
        intervals = [chunk_times[i] - chunk_times[i-1] for i in range(1, len(chunk_times))]
        avg_interval = sum(intervals) / len(intervals)
        min_interval = min(intervals)
        max_interval = max(intervals)
        
        print(f"   Chunk intervals: min={min_interval:.3f}s, max={max_interval:.3f}s, avg={avg_interval:.3f}s")
        
        # Real streaming should have small, variable intervals
        if avg_interval < 0.1 and max_interval < 0.5:
            print("✅ REAL STREAMING: Small, variable intervals indicate genuine streaming")
        else:
            print("⚠️  POSSIBLE SIMULATION: Large intervals suggest batched output")

async def test_langgraph_streaming():
    """Test LangGraph workflow streaming"""
    print("\n🧪 Testing LangGraph Workflow Streaming...")
    
    client = BedrockClient()
    agent = ChatAgent(client)
    
    # Prepare test state
    state = {
        'messages': [HumanMessage(content="What's the weather like in New York? Also calculate 15 * 23.")],
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
    
    config = {"configurable": {"thread_id": "test_chat"}}
    
    print("Starting LangGraph streaming...")
    print("=" * 60)
    
    start_time = time.time()
    chunk_count = 0
    
    try:
        # Test LangGraph streaming
        async for chunk in agent.graph.astream(state, config):
            current_time = time.time()
            chunk_count += 1
            
            print(f"[{current_time - start_time:.2f}s] LangGraph Chunk {chunk_count}:")
            for node_name, node_output in chunk.items():
                print(f"   Node: {node_name}")
                for key, value in node_output.items():
                    if isinstance(value, str) and len(value) > 100:
                        print(f"     {key}: {value[:100]}...")
                    else:
                        print(f"     {key}: {value}")
            print("-" * 40)
    
    except Exception as e:
        print(f"❌ LangGraph Error: {e}")
    
    end_time = time.time()
    print(f"📊 LangGraph completed in {end_time - start_time:.2f}s with {chunk_count} chunks")

def test_reasoning_streaming():
    """Test if reasoning/thinking is actually streamed"""
    print("\n🧪 Testing Reasoning Stream Simulation...")
    
    # This simulates what happens in our current implementation
    full_thought = "I need to think about this complex problem step by step. First, I should understand what the user is asking for. Then I need to break down the problem into smaller parts. After that, I should consider what tools might be helpful. Finally, I need to formulate a comprehensive response."
    
    print("Current 'thinking' emission (instant):")
    print("=" * 60)
    
    start_time = time.time()
    
    # This is what we currently do - emit complete thoughts
    step = f"💭 Reasoning: {full_thought[:100]}..."
    current_time = time.time()
    print(f"[{current_time - start_time:.3f}s] Emitted: {step}")
    
    print("\nWhat REAL reasoning streaming should look like:")
    print("=" * 60)
    
    # This is what real streaming would look like
    words = full_thought.split()
    streamed_text = ""
    
    for i, word in enumerate(words):
        current_time = time.time()
        streamed_text += word + " "
        print(f"[{current_time - start_time:.3f}s] Thinking chunk: '{word}' (total: {len(streamed_text)} chars)")
        time.sleep(0.05)  # Simulate real typing speed

def main():
    """Run all streaming tests"""
    print("🚀 Real-Time Streaming Analysis")
    print("=" * 80)
    
    # Test 1: Raw Bedrock streaming
    test_bedrock_streaming()
    
    # Test 2: LangGraph workflow streaming
    asyncio.run(test_langgraph_streaming())
    
    # Test 3: Reasoning streaming analysis
    test_reasoning_streaming()
    
    print("\n" + "=" * 80)
    print("📋 STREAMING ANALYSIS SUMMARY:")
    print("✅ What IS streaming in real-time:")
    print("   - Final response from Bedrock (token by token)")
    print("   - LangGraph workflow progression (node by node)")
    print("   - WebSocket message transmission")
    
    print("\n❌ What is NOT streaming in real-time:")
    print("   - Reasoning/thinking text (emitted as complete thoughts)")
    print("   - Tool execution results (emitted when complete)")
    print("   - Node-level text generation (only node completion is streamed)")
    
    print("\n🎯 RECOMMENDATION:")
    print("   To achieve true real-time streaming, we need to:")
    print("   1. Stream reasoning text character-by-character during thinking")
    print("   2. Stream individual node responses (reasoning, action planning)")
    print("   3. Stream tool execution progress, not just results")

if __name__ == "__main__":
    main() 