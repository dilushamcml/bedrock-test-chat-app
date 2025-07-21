#!/usr/bin/env python3
"""
Bedrock Chat Application Runner with Poetry
This script runs the Flask application using Poetry's environment
"""

import subprocess
import sys
import os

def run_with_poetry():
    """Run the Flask application using Poetry"""
    try:
        print("🚀 Starting Bedrock Chat Application with Poetry...")
        
        # Change to the project directory
        project_dir = "/home/dilusha/cml/polaris/bedrock_test/streamlit_test_chat_app"
        os.chdir(project_dir)
        
        # Run the application with Poetry
        result = subprocess.run([
            "poetry", "run", "python", "app.py"
        ], check=True)
        
        print("✅ Application stopped.")
        return result.returncode
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running application: {e}")
        return e.returncode
    except KeyboardInterrupt:
        print("\n🛑 Application interrupted by user.")
        return 0
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(run_with_poetry()) 