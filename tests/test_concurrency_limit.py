import asyncio
import httpx
import time

async def trigger_process(i):
    # We'll use a harmless image that exists
    payload = {"path": "/Users/abhishekrai/BO-Decoupled/BO-Video-Tagger/tests/test_media/z-image-1765739783.png", "force_reprocess": True}
    print(f"🚀 Req {i} Sent {time.strftime('%X')}")
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post("http://localhost:8000/process", json=payload)
            print(f"✅ Req {i} Done {time.strftime('%X')} [Status: {resp.status_code}]")
    except Exception as e:
        print(f"❌ Req {i} Failed: {e}")

async def main():
    print("🚦 Testing Concurrency Limit (Expect sequential completion if Semaphore works)")
    # Fire 3 requests at once
    # Since existing logic takes ~1-2s, we should see them finish sequentially roughly 2s apart
    await asyncio.gather(trigger_process(1), trigger_process(2))

if __name__ == "__main__":
    asyncio.run(main())
