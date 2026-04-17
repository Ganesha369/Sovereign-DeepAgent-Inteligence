import os
import json
import asyncio
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
from dotenv import load_dotenv
import litellm
from agent import SovereignAgent

load_dotenv()

app = FastAPI(title="Sovereign Intelligence Engine")
agent = SovereignAgent()

# --- Models ---
class QueryRequest(BaseModel):
    prompt: str
    image_url: Optional[str] = None

class SyntheticTestCase(BaseModel):
    input: str
    expected_output: str

class TestSuiteRequest(BaseModel):
    test_cases: List[SyntheticTestCase]

# --- PILLAR 1: THE SMART ROUTER ---
async def route_query(prompt: str) -> str:
    """
    Heuristic-based router to decide between System 1 (Fast) and System 2 (Deep).
    """
    # Use a small LLM call or regex to decide complexity
    # For production, we use a simple keyword + length check for speed
    complex_keywords = ["research", "browse", "find", "analyze", "plan", "calculate", "steps"]
    is_complex = any(word in prompt.lower() for word in complex_keywords) or len(prompt.split()) > 15
    
    return "SYSTEM_2" if is_complex else "SYSTEM_1"

@app.post("/query")
async def handle_query(request: QueryRequest, response: Response):
    system_type = await route_query(request.prompt)
    response.headers["X-System-Type"] = system_type
    
    if system_type == "SYSTEM_1":
        # System 1: Fast/Cheap using Gemini Flash Lite
        res = await litellm.acompletion(
            model="gemini/gemini-2.0-flash-lite",
            messages=[{"role": "user", "content": request.prompt}]
        )
        return {"system": system_type, "response": res.choices[0].message.content}
    else:
        # System 2: Deep/Reasoning using Sovereign Agent
        res = await agent.run(request.prompt, request.image_url)
        return {"system": system_type, "response": res}

# --- PILLAR 2: SYNTHETIC FACTORY ---
@app.post("/distill")
async def distill_test_cases(request: QueryRequest):
    """
    Uses System 2 to generate 5 high-quality synthetic test cases based on a prompt.
    """
    distill_prompt = f"""
    Act as a Model Distillation expert. Based on the following user query, generate 5 diverse synthetic test cases.
    Each test case must have an 'input' and an 'expected_output'.
    Return ONLY a JSON list of objects.
    
    User Query: {request.prompt}
    """
    # Use System 2 (Deep Reasoning) for high-quality distillation
    raw_json = await agent.run(distill_prompt)
    try:
        # Clean up potential markdown formatting from LLM
        clean_json = raw_json.replace("```json", "").replace("```", "").strip()
        test_cases = json.loads(clean_json)
        return {"synthetic_cases": test_cases}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse synthetic cases: {str(e)}")

# --- PILLAR 3: UNIT TESTER ---
@app.post("/test")
async def run_unit_tests(request: TestSuiteRequest):
    """
    Runs the agent against synthetic cases and uses a Judge LLM to grade results.
    """
    results = []
    for case in request.test_cases:
        # 1. Run Agent
        actual_output = await agent.run(case.input)
        
        # 2. Judge LLM (Gemini Flash)
        judge_prompt = f"""
        Act as an AI Quality Judge.
        Input: {case.input}
        Expected Output: {case.expected_output}
        Actual Output: {actual_output}
        
        Compare the Actual Output against the Expected Output. 
        Does it satisfy the core requirements? 
        Respond with exactly one word: PASS or FAIL.
        """
        judge_res = await litellm.acompletion(
            model="gemini/gemini-2.0-flash-lite",
            messages=[{"role": "user", "content": judge_prompt}]
        )
        grade = judge_res.choices[0].message.content.strip().upper()
        
        results.append({
            "input": case.input,
            "grade": "PASS" if "PASS" in grade else "FAIL",
            "actual": actual_output
        })
        
    return {"test_results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
