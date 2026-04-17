import os
import asyncio
import asyncpg
from dotenv import load_dotenv
import litellm
from playwright.async_api import async_playwright

load_dotenv()

async def test_llm_auth():
    print("--- 1. LiteLLM/Gemini Auth Check ---")
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key or "your_gemini" in api_key:
            print("[SKIP] GEMINI_API_KEY not set.")
            return False
        
        response = await litellm.acompletion(
            model="gemini/gemini-2.0-flash-lite",
            messages=[{"role": "user", "content": "ping"}]
        )
        print(f"[OK] Gemini Auth Successful: {response.choices[0].message.content}")
        return True
    except Exception as e:
        print(f"[FAIL] Gemini Auth Failed: {e}")
        return False

async def test_db_ping():
    print("\n--- 2. Database Ping ---")
    try:
        db_url = os.getenv("DATABASE_URL")
        if not db_url or "your_password" in db_url:
            print("[SKIP] DATABASE_URL not set.")
            return False
            
        conn = await asyncpg.connect(db_url)
        await conn.execute("SELECT 1")
        print("[OK] Database Ping Successful")
        await conn.close()
        return True
    except Exception as e:
        print(f"[FAIL] Database Ping Failed: {e}")
        return False

async def test_browser_launch():
    print("\n--- 3. Playwright Browser Launch Check ---")
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto("https://example.com")
            title = await page.title()
            print(f"[OK] Browser Launched & Navigated: {title}")
            await browser.close()
            return True
    except Exception as e:
        print(f"[FAIL] Browser Launch Failed: {e}")
        return False

async def main():
    print("="*40)
    print("SOVEREIGN INTELLIGENCE 3-POINT CHECK")
    print("="*40)
    
    llm = await test_llm_auth()
    db = await test_db_ping()
    browser = await test_browser_launch()
    
    print("\n" + "="*40)
    if llm and db and browser:
        print("Sovereign Engine Status: [VERIFIED]")
    else:
        print("Sovereign Engine Status: [PARTIAL/FAILED]")
        print(f"LLM: {llm}, DB: {db}, Browser: {browser}")
    print("="*40)

if __name__ == "__main__":
    asyncio.run(main())
