import os
from typing import Optional
import asyncpg
from deepagents import create_deep_agent
from browser_use import Browser
from dotenv import load_dotenv

load_dotenv()

class SovereignAgent:
    def __init__(self):
        # Migrated to 2026 Gemini 3 models
        self.model = "gemini/gemini-3-flash-preview"
        self.db_url = os.getenv("DATABASE_URL")
        
    async def _get_db_conn(self):
        if not self.db_url:
            return None
        try:
            return await asyncpg.connect(self.db_url)
        except Exception as e:
            print(f"⚠️ Warning: Database connection failed: {e}")
            return None

    async def run(self, prompt: str, image_url: Optional[str] = None):
        """
        PILLAR 4: MULTIMODAL SOVEREIGN AGENT
        Uses create_deep_agent pattern with browser tools and Gemini 3.
        """
        # 1. Initialize Browser and Agent
        browser = Browser()
        agent = create_deep_agent(
            model=self.model,
            tools=[browser.get_tool()]
        )
        
        # 2. MULTIMODAL: Construct content
        content = [{"type": "text", "text": prompt}]
        if image_url:
            content.append({
                "type": "image_url", 
                "image_url": {"url": image_url}
            })
            
        # 3. Execute Task
        result = await agent.arun(content)
        
        # 4. PERSISTENCE: Save to PostgreSQL (with error handling)
        await self._persist_result(prompt, str(result))
        
        return result

    async def _persist_result(self, query: str, response: str):
        try:
            conn = await self._get_db_conn()
            if not conn:
                return
            
            # Ensure table exists
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
            await conn.close()
        except Exception as e:
            print(f"⚠️ Warning: Failed to persist result: {e}")
            return
