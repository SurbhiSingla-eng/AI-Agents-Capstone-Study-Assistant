import os
import streamlit as st

from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool
from dotenv import load_dotenv

# ==========================
# Load Environment Variables
# ==========================

load_dotenv()

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

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

if not GEMINI_API_KEY:
    st.error("Missing GEMINI_API_KEY")
    st.stop()

if not SERPER_API_KEY:
    st.error("Missing SERPER_API_KEY")
    st.stop()

# ==========================
# Gemini LLM
# ==========================

llm = LLM(
    model="gemini/gemini-2.5-flash",
    api_key=GEMINI_API_KEY
)

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
# Generate Study Guide
# ==========================

if st.button("Generate Study Guide"):

    if not topic.strip():
        st.warning("Please enter a topic.")
        st.stop()

    with st.spinner("Researching and generating content..."):

        try:

            search_tool = SerperDevTool()

            researcher_agent = Agent(
                role="Senior Web Research Analyst",
                goal=f"Gather the top 5 most recent and comprehensive articles on '{topic}'. Focus on reliable sources.",
                backstory=(
                    "You are an expert researcher skilled at collecting accurate and trustworthy information from the web."
                ),
                tools=[search_tool],
                verbose=False,
                allow_delegation=False,
                llm=llm
            )

            summarizer_agent = Agent(
                role="Academic Content Synthesizer",
                goal="Create a clear, concise, and educational summary.",
                backstory=(
                    "You are an experienced educator who converts complex research into easy-to-understand study material."
                ),
                verbose=False,
                allow_delegation=False,
                llm=llm
            )

            tutor_agent = Agent(
                role="Educational Tutor",
                goal="Generate multiple-choice questions from the study summary.",
                backstory=(
                    "You create high-quality quizzes that help students evaluate their understanding."
                ),
                verbose=False,
                allow_delegation=False,
                llm=llm
            )

            research_task = Task(
                description=(
                    f"Research the topic '{topic}' using the search tool. "
                    "Collect useful facts, concepts, explanations, and recent developments."
                ),
                expected_output="Detailed research notes.",
                agent=researcher_agent
            )

            summarize_task = Task(
                description=(
                    "Create a structured study summary from the research notes. "
                    "Use headings, bullet points, and simple explanations."
                ),
                expected_output="Comprehensive study summary.",
                agent=summarizer_agent,
                context=[research_task]
            )

            quiz_task = Task(
                description=(
                    "Create 5 multiple-choice questions based only on the summary. "
                    "Provide four options and clearly indicate the correct answer."
                ),
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

            result = crew.kickoff()

            st.success("Study Guide Generated Successfully!")

            st.markdown("## Generated Output")
            st.write(result)

        except Exception as e:
            st.error(f"Error: {str(e)}")
