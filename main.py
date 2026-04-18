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
agent_engine = SovereignAgent()

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
    complex_keywords = ["research", "browse", "find", "analyze", "plan", "calculate", "steps"]
    is_complex = any(word in prompt.lower() for word in complex_keywords) or len(prompt.split()) > 15
    return "SYSTEM_2" if is_complex else "SYSTEM_1"

@app.post("/distill")
async def handle_query(request: QueryRequest, response: Response):
    system_type = await route_query(request.prompt)
    response.headers["X-System-Type"] = system_type
    
    if system_type == "SYSTEM_1":
        # System 1: Fast/Reflex using Gemini 3.1 Flash Lite
        res = await litellm.acompletion(
            model="gemini/gemini-3.1-flash-lite-preview",
            messages=[{"role": "user", "content": request.prompt}]
        )
        return {"system": system_type, "response": res.choices[0].message.content}
    else:
        # System 2: Deep/Reasoning - Correctly awaiting agent_engine.run
        res = await agent_engine.run(request.prompt, request.image_url)
        return {"system": system_type, "response": res}

# --- PILLAR 2: SYNTHETIC FACTORY ---
@app.post("/synthetic_distill")
async def distill_test_cases(request: QueryRequest):
    distill_prompt = f"""
    Act as a Model Distillation expert. Based on the following user query, generate 5 diverse synthetic test cases.
    Each test case must have an 'input' and an 'expected_output'.
    Return ONLY a JSON list of objects.
    
    User Query: {request.prompt}
    """
    raw_json = await agent_engine.run(distill_prompt)
    try:
        clean_json = raw_json.replace("```json", "").replace("```", "").strip()
        test_cases = json.loads(clean_json)
        return {"synthetic_cases": test_cases}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse synthetic cases: {str(e)}")

# --- PILLAR 3: UNIT TESTER ---
@app.post("/test")
async def run_unit_tests(request: TestSuiteRequest):
    results = []
    for case in request.test_cases:
        actual_output = await agent_engine.run(case.input)
        
        judge_prompt = f"""
        Act as an AI Quality Judge.
        Input: {case.input}
        Expected Output: {case.expected_output}
        Actual Output: {actual_output}
        
        Compare the Actual Output against the Expected Output. 
        Respond with exactly one word: PASS or FAIL.
        """
        judge_res = await litellm.acompletion(
            model="gemini/gemini-3.1-flash-lite-preview",
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
    # Using Port 8005 for local execution to avoid conflicts
    uvicorn.run(app, host="0.0.0.0", port=8005)
