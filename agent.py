import os
from typing import Optional
import asyncpg
from deepagents import create_deep_agent  # Fixed Import
from browser_use import Browser
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

class SovereignAgent:
    def __init__(self):
        # Initialize the Reasoning Model (System 2)
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-thinking-preview",
            google_api_key=os.getenv("GEMINI_API_KEY")
        )
        self.db_url = os.getenv("DATABASE_URL")
        
    async def _get_db_conn(self):
        if not self.db_url:
            return None
        return await asyncpg.connect(self.db_url)

    async def run(self, prompt: str, image_url: Optional[str] = None):
        """
        PILLAR 4: MULTIMODAL SOVEREIGN AGENT
        Uses create_deep_agent for planning, attaches browser, and persists results.
        """
        # 1. Initialize Browser tool
        browser = Browser()
        browser_tool = browser.get_tool()
        
        # 2. Initialize the actual Deep Agent (The Fix)
        agent = create_deep_agent(
            model=self.llm,
            tools=[browser_tool],
            system_prompt="You are a Sovereign Intelligence Engine. Use the Action Engine for web research."
        )
        
        # 3. MULTIMODAL: Construct messages
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        if image_url:
            messages[0]["content"].append({
                "type": "image_url", 
                "image_url": {"url": image_url}
            })
            
        # 4. Execute Task (Invoke)
        result = await agent.ainvoke({"messages": messages})
        final_output = result["messages"][-1].content
        
        # 5. PERSISTENCE: Save to PostgreSQL
        await self._persist_result(prompt, str(final_output))
        
        return final_output

    async def _persist_result(self, query: str, response: str):
        conn = await self._get_db_conn()
        if not conn:
            return
        
        try:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS agent_research (
                    id SERIAL PRIMARY KEY,
                    query TEXT,
                    response TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            await conn.execute(
                "INSERT INTO agent_research (query, response) VALUES ($1, $2)",
                query, response
            )
        finally:
            await conn.close()