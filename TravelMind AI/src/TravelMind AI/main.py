#!/usr/bin/env python
import logging
import sys
import warnings

from travel_planner_crewai.crew import TravelPlannerCrewai

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("execution.log"),
        ],
    )
    logging.getLogger("LiteLLM").setLevel(logging.CRITICAL)


def _collect_inputs() -> dict:
    destination = input("Destination: ").strip()
    travel_dates = input("Travel dates (e.g. 2026-04-10 to 2026-04-15): ").strip()
    budget = input("Budget (currency + amount): ").strip()
    preferences = input("Preferences (optional, press Enter to skip): ").strip()

    if not destination or not travel_dates or not budget:
        raise ValueError(
            "Missing required travel inputs: destination, travel dates, and budget are required."
        )

    return {
        "destination": destination,
        "travel_dates": travel_dates,
        "budget": budget,
        "preferences": preferences or "No specific preferences provided.",
    }

def run():
    """Run the crew."""
    _configure_logging()
    logger = logging.getLogger(__name__)

    try:
        logger.info("Execution started")
        inputs = _collect_inputs()
        logger.info(
            "Inputs received | destination=%s | travel_dates=%s | budget=%s",
            inputs["destination"],
            inputs["travel_dates"],
            inputs["budget"],
        )
        result = TravelPlannerCrewai().crew().kickoff(inputs=inputs)
        logger.info("Execution completed successfully")
        return result
    except Exception as e:
        logger.exception("Execution failed")
        raise Exception(f"An error occurred while running the crew: {e}")


def train():
    """Train the crew for a given number of iterations."""
    _configure_logging()
    try:
        inputs = _collect_inputs()
        TravelPlannerCrewai().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """Replay the crew execution from a specific task."""
    _configure_logging()
    try:
        TravelPlannerCrewai().crew().replay(task_id=sys.argv[1])
    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """Test the crew execution and returns the results."""
    _configure_logging()

    try:
        inputs = _collect_inputs()
        TravelPlannerCrewai().crew().test(n_iterations=int(sys.argv[1]), eval_llm=sys.argv[2], inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")

def run_with_trigger():
    """Run the crew with trigger payload."""
    _configure_logging()
    import json

    if len(sys.argv) < 2:
        raise Exception("No trigger payload provided. Please provide JSON payload as argument.")

    try:
        trigger_payload = json.loads(sys.argv[1])
    except json.JSONDecodeError:
        raise Exception("Invalid JSON payload provided as argument")

    inputs = _collect_inputs()
    inputs["crewai_trigger_payload"] = trigger_payload

    try:
        result = TravelPlannerCrewai().crew().kickoff(inputs=inputs)
        return result
    except Exception as e:
        raise Exception(f"An error occurred while running the crew with trigger: {e}")
