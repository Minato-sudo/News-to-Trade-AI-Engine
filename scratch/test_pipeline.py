import os
# Must be set before importing any transformers/datasets libs
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import asyncio
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.routes.clusters import background_run_pipeline

async def main():
    print("Starting background pipeline test run with hf-mirror...")
    try:
        await background_run_pipeline(50)
        print("Success!")
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
