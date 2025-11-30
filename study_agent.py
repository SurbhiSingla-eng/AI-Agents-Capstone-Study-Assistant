# ==============================================================================
# STUDENT STUDY & RESEARCH ASSISTANT (Concierge Agent Capstone Project)
# This script defines a three-agent system for automated research, summarization,
# and quiz generation using the Gemini model and CrewAI framework.
# ==============================================================================

# --- CODE BLOCK 1: Installation and Secure Key Setup (For GitHub/Local .env) ---
import os
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool 
from textwrap import dedent
from dotenv import load_dotenv

# Load environment variables from the local .env file
# This assumes the user has a .env file with GEMINI_API_KEY and SERPER_API_KEY
load_dotenv()

# Set environment variables from the loaded .env file
os.environ['GEMINI_API_KEY'] = os.getenv('GEMINI_API_KEY')
os.environ['SERPER_API_KEY'] = os.getenv('SERPER_API_KEY')

# Check for keys (Optional, but good for local debugging)
if not os.environ.get('GEMINI_API_KEY') or not os.environ.get('SERPER_API_KEY'):
    print("FATAL ERROR: API keys not loaded. Ensure .env file is present and keys are defined.")
    exit()

# Set the research topic
research_topic = "The ethical implications of using deepfakes in political campaigns"

# --- CODE BLOCK 2: Agent and Tool Definitions ---

# 1. Define the Tool (Fulfills 'Tools' Requirement)
# The SerperDevTool uses the SERPER_API_KEY to enable internet access.
search_tool = SerperDevTool()

# 2. Define the Agents (Fulfills 'Multi-Agent System' Requirement)

# Agent 1: The Researcher Agent (Tool User)
researcher_agent = Agent(
    role='Senior Web Research Analyst',
    goal=dedent(f"Gather the top 5 most recent and comprehensive articles on: '{research_topic}'. Focus only on credible news or academic sources."),
    backstory=(
        "You are a meticulous and unbiased research analyst known for efficiently sifting "
        "through online data and extracting the most relevant, high-quality information."
    ),
    tools=[search_tool], 
    verbose=True, 
    allow_delegation=False,
    llm='gemini-2.5-flash' # Using Gemini
)

# Agent 2: The Summarizer Agent (LLM Power / Processor)
summarizer_agent = Agent(
    role='Academic Content Synthesizer',
    goal='Condense the raw research text into a concise, accurate, and easy-to-understand summary, strictly 400 words long.',
    backstory=(
        "You are an experienced professor who excels at synthesizing complex information "
        "into clear, paragraph-form study guides, ensuring no critical details are missed."
    ),
    verbose=True,
    allow_delegation=False,
    llm='gemini-2.5-flash'
)

# Agent 3: The Tutor Agent (Memory/Context User / Final Output Generator)
tutor_agent = Agent(
    role='Experienced Educational Tutor',
    goal='Generate 5 unique, challenging multiple-choice questions (MCQs) and their correct answers based ONLY on the summary provided to you.',
    backstory=(
        "You are a dedicated tutor specializing in creating effective practice quizzes "
        "to test comprehension of specific study material. You never use outside information."
    ),
    verbose=True,
    allow_delegation=False,
    llm='gemini-2.5-flash'
)

# --- CODE BLOCK 3: Task Definitions (Sequential Flow) ---

# 1. Research Task (Researcher uses the SerperDevTool)
research_task = Task(
    description=f"Using the SerperDevTool, search for the topic '{research_topic}' and collect the combined text content from the top 5 relevant search results/articles.",
    expected_output="The combined raw text content from the top 5 articles, formatted clearly for the next agent.",
    agent=researcher_agent
)

# 2. Summarize Task (Summarizer uses the output of the Researcher)
summarize_task = Task(
    description="Analyze the raw text content provided in the context and produce a single, professional, 400-word summary suitable for a college-level study guide. Output ONLY the summary text.",
    expected_output="A concise, high-quality, 400-word summary of the research topic.",
    agent=summarizer_agent,
    # Context dependency: Raw research text is passed here (Fulfills 'Memory/Sessions' Requirement)
    context=[research_task] 
)

# 3. Quiz Generation Task (Tutor uses the summary as its context/memory)
quiz_task = Task(
    description="Based STRICTLY on the summary provided in the context, generate 5 challenging multiple-choice questions (MCQs), each with 4 options (A, B, C, D) and the correct answer clearly marked at the end of the question set. Format the output in clean Markdown.",
    expected_output="5 multiple-choice questions in Markdown, followed by an 'Answers:' section listing the correct choices.",
    agent=tutor_agent,
    # Context dependency: Summary is passed here (Fulfills 'Memory/Sessions' Requirement)
    context=[summarize_task] 
)

# --- CODE BLOCK 4: Assemble and Run the Crew ---

if __name__ == "__main__":
    # Define the crew with the sequential process
    study_crew = Crew(
        agents=[researcher_agent, summarizer_agent, tutor_agent],
        tasks=[research_task, summarize_task, quiz_task],
        process=Process.sequential, # Ensures tasks run in order
        verbose=True, # Shows agents' reasoning
    )

    print("--- Starting Study & Research Assistant Multi-Agent System ---")

    # Execute the workflow
    crew_result = study_crew.kickoff()

    print("\n\n########################################")
    print("####### Final Study Guide & Quiz #######")
    print("########################################\n")
    print(crew_result)

# ==============================================================================