from assistant import get_llm_os
import sys
from pathlib import Path

def create_directories():
    """Create necessary directories if they don't exist"""
    Path("tmp/agent_sessions_json").mkdir(parents=True, exist_ok=True)
    Path("tmp").mkdir(parents=True, exist_ok=True)

def initialize_llm_os():
    """Initialize the LLM OS with desired configurations"""
    return get_llm_os(
        calculator=True,  # Enable calculator
        ddg_search=True,  # Enable DuckDuckGo search
        python_assistant=True,  # Enable Python Assistant
        investment_assistant=True,  # Enable Investment Assistant
        shell_tools = True,
        web_crawler = True,
        debug_mode=True,  # Enable debug mode for better visibility
    )

def run_chat_interface():
    """Run the interactive chat interface"""
    print("\n=== LLM OS Chat Interface ===")
    print("Type 'exit' or 'quit' to end the conversation")
    print("Type 'clear' to clear the screen")
    print("================================\n")

    # Create necessary directories
    create_directories()
    
    # Initialize LLM OS
    llm_os = initialize_llm_os()

    while True:
        try:
            # Get user input
            user_input = input("\nYou: ").strip()

            # Handle special commands
            if user_input.lower() in ['exit', 'quit']:
                print("\nGoodbye! Thank you for using LLM OS.")
                sys.exit(0)
            elif user_input.lower() == 'clear':
                # Clear screen - works on both Windows and Unix-like systems
                print('\033[2J\033[H')
                continue
            elif not user_input:
                continue

            # Get response from LLM OS
            response = llm_os.run(user_input)
            
            # Print the response
            if response and response.content:
                print("\nLLM OS:", response.content)
            else:
                print("\nLLM OS: I apologize, but I couldn't generate a response. Please try again.")

        except KeyboardInterrupt:
            print("\n\nExiting gracefully...")
            sys.exit(0)
        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")
            print("Please try again.")

if __name__ == "__main__":
    run_chat_interface()