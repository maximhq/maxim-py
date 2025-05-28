import asyncio
from typing import Coroutine


def run_async(routine: Coroutine):
    """
    Runs a coroutine synchronously.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # If we're in an async context, create a new task and wait for it
        # This is safe in most cases, but if you need the result synchronously, use nest_asyncio or refactor to async
        import nest_asyncio

        nest_asyncio.apply()
        return loop.run_until_complete(routine)
    else:
        return asyncio.run(routine)
