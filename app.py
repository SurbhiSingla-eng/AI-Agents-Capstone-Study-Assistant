import os
import streamlit as st

from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool

# ==========================
# Page Config
# ==========================

st.set_page_config(
    page_title="AI Study Assistant",
    page_icon="📚",
    layout="wide"
)

# ==========================
# Load Secrets
# ==========================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

if not GEMINI_API_KEY:
    st.error("Missing GEMINI_API_KEY secret.")
    st.stop()

if not SERPER_API_KEY:
    st.error("Missing SERPER_API_KEY secret.")
    st.stop()

# ==========================
# Gemini LLM
# ==========================

llm = LLM(
    model="gemini/gemini-1.5-flash",
    api_key=GEMINI_API_KEY
)

# ==========================
# UI
# ==========================

st.title("📚 AI Study Assistant")

st.markdown("""
Generate:
- Research Notes
- Study Summary
- MCQ Quiz

using a multi-agent AI workflow powered by CrewAI and Gemini.
""")

topic = st.text_input(
    "Enter a study topic",
    placeholder="Machine Learning"
)

# ==========================
# Generate Button
# ==========================

if st.button("Generate Study Guide"):

    if not topic.strip():
        st.warning("Please enter a topic.")
        st.stop()

    try:

        with st.spinner("Researching topic..."):

            search_tool = SerperDevTool()

            researcher_agent = Agent(
                role="Senior Web Research Analyst",
                goal=f"Research {topic} thoroughly using reliable web sources.",
                backstory=(
                    "You are an expert researcher who gathers accurate and up-to-date information."
                ),
                tools=[search_tool],
                allow_delegation=False,
                verbose=False,
                llm=llm
            )

            summarizer_agent = Agent(
                role="Academic Content Synthesizer",
                goal="Convert research into a clear study summary.",
                backstory=(
                    "You are an experienced educator who simplifies complex concepts."
                ),
                allow_delegation=False,
                verbose=False,
                llm=llm
            )

            tutor_agent = Agent(
                role="Educational Tutor",
                goal="Create multiple-choice questions from the study summary.",
                backstory=(
                    "You create quizzes that help students test their understanding."
                ),
                allow_delegation=False,
                verbose=False,
                llm=llm
            )

            research_task = Task(
                description=(
                    f"Research the topic '{topic}'. "
                    "Collect key concepts, definitions, examples, advantages, disadvantages, and recent developments."
                ),
                expected_output="Detailed research notes.",
                agent=researcher_agent
            )

            summary_task = Task(
                description=(
                    "Create a structured study summary using headings, bullet points, and concise explanations."
                ),
                expected_output="A complete study summary.",
                agent=summarizer_agent,
                context=[research_task]
            )

            quiz_task = Task(
                description=(
                    "Create 5 multiple-choice questions based only on the study summary. "
                    "Provide four options and clearly indicate the correct answer."
                ),
                expected_output="5 MCQs with answers.",
                agent=tutor_agent,
                context=[summary_task]
            )

            crew = Crew(
                agents=[
                    researcher_agent,
                    summarizer_agent,
                    tutor_agent
                ],
                tasks=[
                    research_task,
                    summary_task,
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
