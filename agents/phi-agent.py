from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv
from phi.tools.yfinance import YFinanceTools
import os
import phi
from phi.playground import Playground,serve_playground_app
from phi.agent import Agent
from phi.model.openai import OpenAIChat

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
phi.ap = os.getenv("PHI_API_KEY")

web_agent = Agent(
    name="Web Search Agent",
    model=OpenAIChat(id="gpt-4o"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tool_calls=True,
    markdown=True,
)

finance_agent = Agent(
    name="Finance Agent",
    model=OpenAIChat(model="gpt-4o-mini"),  # or gpt-4.1, gpt-4o, etc.
    tools=[
        YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            company_info=True,
            historical_prices=True,
            news=True,
        )
    ],
    instructions="You are a helpful financial assistant."
)

app = Playground(agents=[web_agent, finance_agent]).get_app()

if __name__ == "__main__":
    serve_playground_app("playground:app", reload=True)