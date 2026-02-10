from typing import TypedDict, Literal,Annotated
from langchain_community.chat_models import ChatOllama
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage
import operator
from pydantic import BaseModel, Field
import json
import re

# Initialize the Ollama model
generator_model = ChatOllama(model="llama3.1", temperature=0)
evaluator_model = ChatOllama(model="llama3.1", temperature=0)
optimizer_model = ChatOllama(model="llama3.1", temperature=0)

class TweetEvaluation(BaseModel):
    evaluation: Literal["approved", "needs_improvement"] = Field(..., description="Final evaluation result.")
    feedback: str = Field(..., description="feedback for the tweet.")


def parse_evaluator_response(response_text: str) -> dict:
    """Parse evaluator responses robustly.

    Handles JSON, embedded JSON, YAML-like key:value pairs, and free text.
    Returns a dict with keys 'evaluation' and 'feedback'.
    """
    if not response_text or not response_text.strip():
        return {"evaluation": "needs_improvement", "feedback": "Evaluator returned an empty response."}

    # Try JSON
    try:
        parsed = json.loads(response_text)
        if isinstance(parsed, dict) and 'evaluation' in parsed and 'feedback' in parsed:
            return parsed
    except Exception:
        pass

    # Try to extract first JSON object
    m = re.search(r"\{.*\}", response_text, re.S)
    if m:
        try:
            parsed = json.loads(m.group(0))
            if isinstance(parsed, dict) and 'evaluation' in parsed and 'feedback' in parsed:
                return parsed
        except Exception:
            pass

    # Parse simple key: value lines
    kv = {}
    for line in response_text.splitlines():
        if ':' in line:
            k, v = line.split(':', 1)
            k = k.strip().strip('"').strip("'").lower()
            v = v.strip().strip('"').strip("'")
            kv[k] = v
    if 'evaluation' in kv and 'feedback' in kv:
        return {'evaluation': kv['evaluation'], 'feedback': kv['feedback']}

    # Fallback heuristics
    eval_m = re.search(r"\b(approved|needs_improvement|needs improvement|needs-improvement)\b", response_text, re.I)
    if eval_m:
        ev = eval_m.group(1).lower()
        if 'needs' in ev:
            evaluation = 'needs_improvement'
        else:
            evaluation = 'approved'
    else:
        evaluation = 'needs_improvement'

    fb_m = re.search(r'feedback[:\-]\s*(.*)', response_text, re.I | re.S)
    if fb_m:
        feedback = fb_m.group(1).strip()
    else:
        # Use the entire response as feedback if nothing else
        feedback = response_text.strip()

    return {'evaluation': evaluation, 'feedback': feedback}

# Define the state structure
class TweetState(TypedDict):
    topic: str
    tweet: str
    evaluation: Literal["approved", "needs_improvement"]
    feedback: str
    iteration: int
    max_iteration: int
    
    tweet_history: Annotated[list[str], operator.add]
    feedback_history: Annotated[list[str], operator.add]

def generate_tweet(state: TweetState):
    # prompt
    messages = [
        SystemMessage(content="You are a funny and clever Twitter/X influencer."),
        HumanMessage(content=f"""
Write a short, original, and hilarious tweet on the topic: "{state['topic']}".

Rules:
- Do NOT use question-answer format.
- Max 280 characters.
- Use observational humor, irony, sarcasm, or cultural references.
- Think in meme logic, punchlines, or relatable takes.
- Use simple, day to day english
""")
    ]

    # send generator_llm
    response = generator_model.invoke(messages).content

    # return response
    return {'tweet': response,'tweet_history': [response]}


def evaluate_tweet(state: TweetState):

    # prompt
    messages = [
    SystemMessage(content="You are a ruthless, no-laugh-given Twitter critic. You evaluate tweets based on humor, originality, virality, and tweet format."),
    HumanMessage(content=f"""
Evaluate the following tweet:

Tweet: "{state['tweet']}"

Use the criteria below to evaluate the tweet:

1. Originality – Is this fresh, or have you seen it a hundred times before?  
2. Humor – Did it genuinely make you smile, laugh, or chuckle?  
3. Punchiness – Is it short, sharp, and scroll-stopping?  
4. Virality Potential – Would people retweet or share it?  
5. Format – Is it a well-formed tweet (not a setup-punchline joke, not a Q&A joke, and under 280 characters)?

Auto-reject if:
- It's written in question-answer format (e.g., "Why did..." or "What happens when...")
- It exceeds 280 characters
- It reads like a traditional setup-punchline joke
- Dont end with generic, throwaway, or deflating lines that weaken the humor (e.g., “Masterpieces of the auntie-uncle universe” or vague summaries)

### Respond ONLY in structured format:
- evaluation: "approved" or "needs_improvement"  
- feedback: One paragraph explaining the strengths and weaknesses 
""")
]

    response_text = evaluator_model.invoke(messages).content

    parsed = parse_evaluator_response(response_text)

    # Normalize evaluation to expected values
    if parsed.get('evaluation') not in ("approved", "needs_improvement"):
        parsed['evaluation'] = "needs_improvement"

    return {'evaluation': parsed['evaluation'], 'feedback': parsed['feedback'], 'feedback_history': [parsed['feedback']]}
def optimize_tweet(state: TweetState):

    messages = [
        SystemMessage(content="You punch up tweets for virality and humor based on given feedback."),
        HumanMessage(content=f"""
Improve the tweet based on this feedback:
"{state['feedback']}"

Topic: "{state['topic']}"
Original Tweet:
{state['tweet']}

Re-write it as a short, viral-worthy tweet. Avoid Q&A style and stay under 280 characters.
""")
    ]

    response = optimizer_model.invoke(messages).content
    iteration = state['iteration'] + 1

    return {'tweet': response, 'iteration': iteration, 'tweet_history': [response]}

def route_evaluation(state:TweetState):
   if state['evaluation'] == 'approved' or state['iteration'] >= state['max_iteration']:
        return 'approved'
   else:
        return 'needs_improvement'

graph= StateGraph(TweetState)
graph.add_node("generate_tweet", generate_tweet)
graph.add_node("evaluate_tweet", evaluate_tweet)
graph.add_node("optimize_tweet", optimize_tweet)
graph.add_edge(START, "generate_tweet")
graph.add_edge("generate_tweet", "evaluate_tweet")
graph.add_conditional_edges("evaluate_tweet", 
                            route_evaluation,
                            {"approved": END, "needs_improvement": "optimize_tweet"})
graph.add_edge("optimize_tweet", "evaluate_tweet")

workflow = graph.compile()

initial_state = {
    "topic": "2026 Budget Analysis & sectors impacted",
    "iteration": 1,
    "max_iteration": 5
}
result = workflow.invoke(initial_state)
print(result)

for tweet in result['tweet_history']:
    print(tweet)

