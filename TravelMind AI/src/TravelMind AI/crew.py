import logging
import os
from typing import List

from crewai import Agent, Crew, LLM, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task

from travel_planner_crewai.tools import SerperTravelSearchTool, TravelBudgetCalculatorTool

logger = logging.getLogger(__name__)

@CrewBase
class TravelPlannerCrewai():
    """TravelPlannerCrewai crew."""

    agents: List[BaseAgent]
    tasks: List[Task]

    def _groq_llm(self) -> LLM:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY is required to run this crew.")

        model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        if not model.startswith("groq/"):
            model = f"groq/{model}"

        return LLM(model=model, api_key=api_key)

    @agent
    def destination_researcher(self) -> Agent:
        logger.info("Initializing destination_researcher")
        return Agent(
            config=self.agents_config['destination_researcher'],  # type: ignore[index]
            llm=self._groq_llm(),
            tools=[SerperTravelSearchTool()],
            verbose=True,
        )

    @agent
    def budget_planner(self) -> Agent:
        logger.info("Initializing budget_planner")
        return Agent(
            config=self.agents_config['budget_planner'],  # type: ignore[index]
            llm=self._groq_llm(),
            tools=[SerperTravelSearchTool(), TravelBudgetCalculatorTool()],
            verbose=True,
        )

    @task
    def destination_research_task(self) -> Task:
        logger.info("Creating destination_research_task")
        return Task(
            config=self.tasks_config['destination_research_task'],  # type: ignore[index]
        )

    @task
    def budget_planning_task(self) -> Task:
        logger.info("Creating budget_planning_task")
        return Task(
            config=self.tasks_config['budget_planning_task'],  # type: ignore[index]
            context=[self.destination_research_task()],
        )

    @agent
    def itinerary_designer(self) -> Agent:
        logger.info("Initializing itinerary_designer")
        return Agent(
            config=self.agents_config['itinerary_designer'],  # type: ignore[index]
            llm=self._groq_llm(),
            tools=[SerperTravelSearchTool()],
            verbose=True,
        )

    @agent
    def validation_agent(self) -> Agent:
        logger.info("Initializing validation_agent")
        return Agent(
            config=self.agents_config['validation_agent'],  # type: ignore[index]
            llm=self._groq_llm(),
            verbose=True,
        )

    @task
    def itinerary_design_task(self) -> Task:
        logger.info("Creating itinerary_design_task")
        return Task(
            config=self.tasks_config['itinerary_design_task'],  # type: ignore[index]
            context=[self.destination_research_task(), self.budget_planning_task()],
        )

    @task
    def validation_task(self) -> Task:
        logger.info("Creating validation_task")
        return Task(
            config=self.tasks_config['validation_task'],  # type: ignore[index]
            context=[
                self.destination_research_task(),
                self.budget_planning_task(),
                self.itinerary_design_task(),
            ],
            output_file='report.md',
        )

    @crew
    def crew(self) -> Crew:
        """Creates the TravelPlannerCrewai crew."""
        logger.info("Creating TravelPlannerCrewai with sequential process")

        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
