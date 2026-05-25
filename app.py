import os
import streamlit as st

from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
from dotenv import load_dotenv

# ==========================
# Load Environment Variables
# ==========================

load_dotenv()

os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY", "")
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY", "")

# ==========================
# Streamlit Page Config
# ==========================

st.set_page_config(
    page_title="AI Study Assistant",
    page_icon="📚",
    layout="wide"
)

# ==========================
# API Key Validation
# ==========================

if not os.getenv("GEMINI_API_KEY"):
    st.error("Missing GEMINI_API_KEY")
    st.stop()

if not os.getenv("SERPER_API_KEY"):
    st.error("Missing SERPER_API_KEY")
    st.stop()

# ==========================
# UI
# ==========================

st.title("📚 AI Study Assistant")

st.write(
    """
    Generate research summaries and quizzes using a multi-agent AI system.
    """
)

topic = st.text_input(
    "Enter a study topic",
    placeholder="Machine Learning"
)

# ==========================
# Run Crew
# ==========================

if st.button("Generate Study Guide"):

    if not topic.strip():
        st.warning("Please enter a topic.")
        st.stop()

    with st.spinner("Researching and generating content..."):

        search_tool = SerperDevTool()

        researcher_agent = Agent(
            role="Senior Web Research Analyst",
            goal=f"Gather the top 5 most recent and comprehensive articles on '{topic}'. Focus on credible sources.",
            backstory=(
                "You are an expert web researcher who collects accurate and reliable information."
            ),
            tools=[search_tool],
            verbose=False,
            allow_delegation=False,
            llm="gemini-2.5-flash"
        )

        summarizer_agent = Agent(
            role="Academic Content Synthesizer",
            goal="Create a clear and concise study summary.",
            backstory=(
                "You are an experienced educator who converts complex topics into easy-to-understand summaries."
            ),
            verbose=False,
            allow_delegation=False,
            llm="gemini-2.5-flash"
        )

        tutor_agent = Agent(
            role="Educational Tutor",
            goal="Generate multiple-choice questions from the summary.",
            backstory=(
                "You create high-quality quizzes to evaluate understanding of study material."
            ),
            verbose=False,
            allow_delegation=False,
            llm="gemini-2.5-flash"
        )

        research_task = Task(
            description=f"Research the topic '{topic}' using the search tool and gather useful information.",
            expected_output="Comprehensive research notes.",
            agent=researcher_agent
        )

        summarize_task = Task(
            description="Create a detailed and well-structured study summary from the research notes.",
            expected_output="A complete summary.",
            agent=summarizer_agent,
            context=[research_task]
        )

        quiz_task = Task(
            description="Generate 5 MCQs based only on the summary.",
            expected_output="5 MCQs with answers.",
            agent=tutor_agent,
            context=[summarize_task]
        )

        crew = Crew(
            agents=[
                researcher_agent,
                summarizer_agent,
                tutor_agent
            ],
            tasks=[
                research_task,
                summarize_task,
                quiz_task
            ],
            process=Process.sequential,
            verbose=False
        )

        try:
            result = crew.kickoff()

            st.success("Study Guide Generated Successfully!")

            st.markdown("## Generated Output")
            st.write(result)

        except Exception as e:
            st.error(f"Error: {str(e)}")
