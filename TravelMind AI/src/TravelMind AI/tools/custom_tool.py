import ast
import logging
import operator as op
from typing import Any, Type

from crewai.tools import BaseTool
from crewai_tools import SerperDevTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_ALLOWED_OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
}


def calculate_expression(expression: str) -> float:
    """Safely evaluate basic arithmetic expressions for travel budget calculations."""

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
            return _ALLOWED_OPERATORS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPERATORS:
            return _ALLOWED_OPERATORS[type(node.op)](_eval(node.operand))
        raise ValueError("Unsupported expression. Use only numbers and + - * / ** operators.")

    parsed = ast.parse(expression, mode="eval")
    return _eval(parsed.body)


class SerperSearchInput(BaseModel):
    query: str = Field(..., description="Travel-related web query to send to Serper.")


class SerperTravelSearchTool(BaseTool):
    name: str = "serper_travel_search"
    description: str = (
        "Search the web for current destination highlights, pricing, and travel logistics using Serper Dev API."
    )
    args_schema: Type[BaseModel] = SerperSearchInput

    def _run(self, query: str) -> str:
        logger.info("Serper search started | query=%s", query)
        result = SerperDevTool()._run(search_query=query)
        logger.info("Serper search finished")
        return str(result)


class BudgetCalculatorInput(BaseModel):
    expression: str = Field(
        ...,
        description="Arithmetic expression for budget computation, e.g. '(120*4)+(35*5)+90+150'.",
    )


class TravelBudgetCalculatorTool(BaseTool):
    name: str = "travel_budget_calculator"
    description: str = (
        "Evaluate arithmetic expressions for budget totals and subtotals."
    )
    args_schema: Type[BaseModel] = BudgetCalculatorInput

    def _run(self, expression: str) -> str:
        logger.info("Budget calculation started | expression=%s", expression)
        value = calculate_expression(expression)
        logger.info("Budget calculation finished | total=%s", value)
        return str(value)
