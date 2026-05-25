import os
import streamlit as st

from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
from textwrap import dedent
from dotenv import load_dotenv

load_dotenv()

# Load API Keys
os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")

st.set_page_config(
    page_title="AI Study Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 AI Study Assistant")
st.write(
    "Generate research summaries and quizzes using a multi-agent AI system."
)

topic = st.text_input(
    "Enter a study topic",
    placeholder="Machine Learning"
)

if st.button("Generate Study Guide"):

    if not topic.strip():
        st.warning("Please enter a topic.")
        st.stop()

    with st.spinner("Researching and generating content..."):

        search_tool = SerperDevTool()

        researcher_agent = Agent(
            role="Senior Web Research Analyst",
            goal=f"Gather the top 5 most recent and comprehensive articles on: '{topic}'",
            backstory="Expert web researcher.",
            tools=[search_tool],
            verbose=False,
            allow_delegation=False,
            llm="gemini-2.5-flash"
        )

        summarizer_agent = Agent(
            role="Academic Content Synthesizer",
            goal="Create a concise academic summary.",
            backstory="Expert educator.",
            verbose=False,
            allow_delegation=False,
            llm="gemini-2.5-flash"
        )

        tutor_agent = Agent(
            role="Educational Tutor",
            goal="Generate MCQs from the summary.",
            backstory="Creates educational quizzes.",
            verbose=False,
            allow_delegation=False,
            llm="gemini-2.5-flash"
        )

        research_task = Task(
            description=f"Research '{topic}' using the search tool.",
            expected_output="Raw research notes.",
            agent=researcher_agent
        )

        summarize_task = Task(
            description="Create a detailed summary.",
            expected_output="Summary.",
            agent=summarizer_agent,
            context=[research_task]
        )

        quiz_task = Task(
            description="Create 5 MCQs from the summary.",
            expected_output="Quiz with answers.",
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

    st.success("Done!")

    st.markdown("## Generated Output")
    st.write(result)
