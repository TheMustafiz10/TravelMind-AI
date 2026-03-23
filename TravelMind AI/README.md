# Multi-Agent AI Travel Planner (CrewAI)

This project implements a CrewAI-based travel planner with the required architecture and tooling:

- Agents: Destination Researcher, Budget Planner, Itinerary Designer, Validation Agent
- Mandatory search: Serper Dev API
- LLM connectivity: Groq API
- Custom calculator logic for budget computations
- Structured markdown output in `report.md`
- Execution logging in `execution.log`

## Input and Output

### User input

- Destination
- Travel Dates
- Budget
- Preferences (Optional)

### Output sections

1. Travel Plan: `<Destination>`
2. Destination Overview
3. Budget Breakdown
4. Day-wise Itinerary
5. Validation Summary

## Project Structure

```
TravelMind AI/
├── src/
│   └── TravelMind AI/
│       ├── __init__.py
│       ├── crew.py
│       ├── main.py
│       ├── config/
│       │   ├── agents.yaml
│       │   └── tasks.yaml
│       └── tools/
│           ├── __init__.py
│           └── custom_tool.py
├── pyproject.toml
├── report.md
└── README.md
```

## Setup

Python requirement: `>=3.10,<3.14`

Clone the repository:

```bash
cd TravelMind AI
```

Install dependencies:

```bash
crewai install
```

Set environment variables:

```bash
export GROQ_API_KEY="your_groq_api_key"
export SERPER_API_KEY="your_serper_api_key"
export GROQ_MODEL="llama-3.3-70b-versatile"  # optional
```

## Run

```bash
crewai run
```

The CLI prompts for destination, travel dates, budget, and optional preferences.

## Analysis

**Why multi-agent?** Each agent focuses on a single responsibility (research, budgeting, itinerary design, validation) with its own tools and prompt. This improves accuracy compared to a single LLM call and keeps each context window small and task-specific.

**What if Serper returns incorrect data?** Serper results are not guaranteed to be accurate. The Validation Agent flags unverifiable claims as assumptions, but incorrect data can still propagate; users should verify critical details before travel.

**What if budget is unrealistic?** The Budget Planner explicitly flags any shortfall rather than forcing a plan to fit. The Validation Agent surfaces budget warnings in the final summary.

**Hallucination risks?** LLMs can fabricate plausible details. Grounding via live Serper results and a dedicated Validation Agent reduce this, but cannot eliminate it entirely. Unverified data points are marked as assumptions in the report.

**Token usage?** Per-agent prompts keep each call lightweight, but context accumulates across the workflow. The `llama-3.3-70b-versatile` 128k window is enough for most trips, though very long trips may approach the limit.

**Scalability?** The current design is single-user and stateless. Multi-user scaling would require isolated crew instances per request, queuing, and caching. Higher trip complexity directly increases token and Serper API usage.
