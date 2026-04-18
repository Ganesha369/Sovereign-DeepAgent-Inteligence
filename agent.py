import os
from typing import Optional
import asyncpg
from browser_use import Browser, Agent
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

class BrowserCompatibleLLM:
    def __init__(self, llm):
        self.llm = llm
        self.provider = "google"
    
    def __getattr__(self, name):
        return getattr(self.llm, name)

class SovereignAgent:
    def __init__(self):
        # Migrated to 2026 Gemini 3 models
        self.model = "gemini-3-flash-preview"
        self.db_url = os.getenv("DATABASE_URL")
        real_llm = ChatGoogleGenerativeAI(
            model=self.model, 
            api_key=os.getenv("GEMINI_API_KEY")
        )
        self.llm = BrowserCompatibleLLM(real_llm)
        self.browser_session = Browser()
        
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
        Uses Agent from browser_use with Gemini 3.
        """
        # 1. Initialize Agent with browser object directly
        self.agent = Agent(
            task=prompt,
            llm=self.llm,
            browser=self.browser_session
        )
        
        # 2. Execute Task
        result = await self.agent.run()
        
        # 3. PERSISTENCE: Save to PostgreSQL (with error handling)
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
