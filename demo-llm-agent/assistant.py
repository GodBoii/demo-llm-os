import json
from pathlib import Path
from typing import Optional
from textwrap import dedent
from typing import List
from phi.tools import Toolkit
from phi.tools.shell import ShellTools
from phi.tools.calculator import Calculator
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools
from phi.tools.python import PythonTools
from phi.tools.crawl4ai_tools import Crawl4aiTools
from phi.tools.website import WebsiteTools
from phi.utils.log import logger
from phi.agent import Agent, AgentMemory
from phi.memory.classifier import MemoryClassifier
from phi.memory.summarizer import MemorySummarizer
from phi.model.google import Gemini
from phi.model.groq import Groq
from phi.memory.db.sqlite import SqliteMemoryDb
from phi.storage.agent.json import JsonFileAgentStorage

class CustomJsonFileAgentStorage(JsonFileAgentStorage):
    def serialize(self, data: dict) -> str:
        # Clean up Gemini's parts before serialization
        if data.get("agent_data", {}).get("model", {}).get("provider") == "Google":
            if "memory" in data:
                # Clean up runs' response messages
                if "runs" in data["memory"]:
                    for run in data["memory"]["runs"]:
                        if "response" in run and "messages" in run["response"]:
                            for m in run["response"]["messages"]:
                                if isinstance(m, dict):
                                    m.pop("parts", None)
                
                # Clean up top-level memory messages
                if "messages" in data["memory"]:
                    for m in data["memory"]["messages"]:
                        if isinstance(m, dict):
                            m.pop("parts", None)
        
        return super().serialize(data)

def get_llm_os(
    llm_id: str = "gemini-2.0-flash-exp",
    calculator: bool = False,
    web_crawler: bool = False,
    ddg_search: bool = False,
    shell_tools: bool = False,
    python_assistant: bool = False,
    investment_assistant: bool = False,
    user_id: Optional[str] = None,
    run_id: Optional[str] = None,
    debug_mode: bool = True,
) -> Agent:
    logger.info(f"-*- Creating {llm_id} LLM OS -*-")

    # Add tools available to the LLM OS
    tools: List[Toolkit] = []
    extra_instructions: List[str] = []

    if calculator:
        calc_tool = Calculator(
            add=True,
            subtract=True,
            multiply=True,
            divide=True,
            exponentiate=True,
            factorial=True,
            is_prime=True,
            square_root=True,
        )
        tools.append(calc_tool)
        extra_instructions.append(
            "Use the Calculator tool for mathematical operations. Available functions: add, subtract, multiply, divide, exponentiate, factorial, is_prime, square_root"
        )

    if web_crawler:
        crawl_tool = Crawl4aiTools(max_length=None)
        tools.append(crawl_tool)
        extra_instructions.append(
            "Use the web_crawler tool when any URL is provided. Example: web_crawler(url='https://example.com')"
        )

    if ddg_search:
        ddg_tool = DuckDuckGo(fixed_max_results=10)
        tools.append(ddg_tool)
        extra_instructions.append(
            "Use the DuckDuckGo search tool to find current information from the internet. Example: duckduckgo_search(query='your search query') and Always include sources"
        )
    
    if shell_tools:
        shell_tool = ShellTools()
        tools.append(shell_tool)
        extra_instructions.append(
            "Use the shell_tools for system and file operations. Example: run_shell_command(args='ls -la') for directory contents"
        )

    team: List[Agent] = []
    if python_assistant:
        _python_assistant = Agent(
            name="Python Assistant",
            tools=[PythonTools()],
            role="Write and run python code",
            model=Groq(id="llama-3.3-70b-versatile"),
            pip_install=True,
        )
        team.append(_python_assistant)
        extra_instructions.append("To write and run python code, delegate the task to the `Python Assistant`.")

    if investment_assistant:
        _investment_assistant = Agent(
            name="Investment Assistant",
            role="Write a investment report on a given company (stock) symbol",
            model=Gemini(id="gemini-2.0-flash-exp"),
            description="You are a Senior Investment Analyst for Goldman Sachs tasked with writing an investment report for a very important client.",
            instructions=[
                "For a given stock symbol, get the stock price, company information, analyst recommendations, and company news",
                "Carefully read the research and generate a final - Goldman Sachs worthy investment report in the <report_format> provided below.",
                "Provide thoughtful insights and recommendations based on the research.",
                "When you share numbers, make sure to include the units (e.g., millions/billions) and currency.",
                "REMEMBER: This report is for a very important client, so the quality of the report is important.",
            ],
            expected_output=dedent(
                """\
            <report_format>
            ## [Company Name]: Investment Report

            ### **Overview**
            {give a brief introduction of the company and why the user should read this report}
            {make this section engaging and create a hook for the reader}

            ### Core Metrics
            {provide a summary of core metrics and show the latest data}
            - Current price: {current price}
            - 52-week high: {52-week high}
            - 52-week low: {52-week low}
            - Market Cap: {Market Cap} in billions
            - P/E Ratio: {P/E Ratio}
            - Earnings per Share: {EPS}
            - 50-day average: {50-day average}
            - 200-day average: {200-day average}
            - Analyst Recommendations: {buy, hold, sell} (number of analysts)

            ### Financial Performance
            {analyze the company's financial performance}

            ### Growth Prospects
            {analyze the company's growth prospects and future potential}

            ### News and Updates
            {summarize relevant news that can impact the stock price}

            ### [Summary]
            {give a summary of the report and what are the key takeaways}

            ### [Recommendation]
            {provide a recommendation on the stock along with a thorough reasoning}

            </report_format>
            """
            ),
            tools=[YFinanceTools(stock_price=True, company_info=True, analyst_recommendations=True, company_news=True)],
            # This setting tells the LLM to format messages in markdown
            markdown=True,
            add_datetime_to_instructions=True,
            debug_mode=debug_mode,
        )
        team.append(_investment_assistant)
        extra_instructions.extend(
            [
                "To get an investment report on a stock, delegate the task to the `Investment Assistant`. "
                "Return the report in the <report_format> to the user without any additional text like 'here is the report'.",
                "Answer any questions they may have using the information in the report.",
                "Never provide investment advise without the investment report.",
            ]
        )

    llm_os = Agent(
        name="AI_OS",
        run_id=run_id,
        user_id=user_id,
        model=Gemini(id="gemini-2.0-flash-exp"),
        #model=Groq(id="llama-3.3-70b-versatile"),
        description=dedent(
            """\
        You are the most advanced AI system in the world called `AI-OS`.
        You have access to a set of tools and a team of AI Assistants at your disposal.
        You must use the appropriate tool for each task:
        Your goal is to assist the user in the best way possible.\
        """
        ),
        instructions=[
            "When the user sends a message, first **think** and determine if:\n"
            " - You can answer by using a tool available to you\n"
            " - You need to search the knowledge base\n"
            " - You need to search the internet\n"
            " - You need to delegate the task to a team member\n"
            " - You need to ask a clarifying question",
            "For mathematical calculations, you can use the Calculator tool if you think you won't be able to provide accurate answer",
            "For internet searches, ALWAYS use the DuckDuckGo search tool to get current information",
            "For system operations, use the shell tools when necessary",
            "If the user asks about a topic, first ALWAYS search your knowledge base using the `search_knowledge_base` tool.",
            "If you dont find relevant information in your knowledge base, use the `duckduckgo_search` tool to search the internet.",
            "If the user asks to summarize the conversation or if you need to reference your chat history with the user, use the `get_chat_history` tool.",
            "If the users message is unclear, ask clarifying questions to get more information.",
            "Carefully read the information you have gathered and provide a clear and concise answer to the user.",
            "Do not use phrases like 'based on my knowledge' or 'depending on the information'.",
            "You can delegate tasks to an AI Assistant in your team depending of their role and the tools available to them.",
            "When a user provides a URL, IMMEDIATELY use the web_crawler tool without any preliminary message",
            "When a user asks about files, directories, or system information, IMMEDIATELY use shell_tools without any preliminary message",
            "Do not explain what you're going to do - just use the appropriate tool right away",        
        ],  
        extra_instructions=extra_instructions,
        # Add long-term memory to the LLM OS backed by a PostgreSQL database
        storage=CustomJsonFileAgentStorage(dir_path="tmp/agent_sessions_json"),
        memory=AgentMemory(
            classifier=MemoryClassifier(model=Groq(id="llama-3.3-70b-versatile")),
            summarizer=MemorySummarizer(model=Groq(id="llama-3.3-70b-versatile")),
            db=SqliteMemoryDb(
                table_name="agent_memory",
                db_file="tmp/agent_memory.db",
            ),
            create_user_memories=True,
            update_user_memories_after_run=True,
            create_session_summary=True,
            update_session_summary_after_run=True,
        ),
        # Add a knowledge base to the LLM OS
        # Add selected tools to the LLM OS
        tools=tools,
        # Add selected team members to the LLM OS
        team=team,
        # Show tool calls in the chat
        show_tool_calls=True,
        # This setting gives the LLM a tool to search the knowledge base for information
        tool_choice="auto",    # Important: This allows the model to choose when to use tools
        search_knowledge=True,
        # This setting gives the LLM a tool to get chat history
        read_chat_history=True,
        # This setting adds chat history to the messages
        add_chat_history_to_messages=True,
        # This setting adds 6 previous messages from chat history to the messages sent to the LLM
        num_history_messages=6,
        # This setting tells the LLM to format messages in markdown
        markdown=True,
        # This setting adds the current datetime to the instructions
        add_datetime_to_instructions=True,
        # Add an introductory Assistant message
        introduction=dedent(
            """\
        Hi, I'm your LLM OS.
        I have access to a set of tools and AI Assistants to assist you.
        Let's solve some problems together!\
        """
        ),
        debug_mode=debug_mode,
    )
    return llm_os