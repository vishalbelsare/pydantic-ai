import asyncio
import random

import httpx

mock_user_prompts = realistic_engineer_prompts = [
    "last 24 hours",
    "2 hours ago to now",
    "today 09:00 to 13:00 local",
    "yesterday 7 AM to 7 PM",
    "around the crash at 14:05 UTC on Mar 3",
    "August 10",
    "yesterday 2 PM to 3:30 PM local time",
    "all logs from Feb 5 PST",
    "7 AM to 2 PM on Monday",
    "the past 6 hours",
    "last 15 minutes",
    "noon to 2 PM Thursday",
    "Jan 12, 6 AM to noon",
    "March 1, 00:00 to 23:59 local",
    "10:00 UTC to 16:00 UTC on May 10",
    "the spike at 4:17 PM local on Mar 3",
    "all data from last Friday morning to last Friday evening",
    "from last Monday morning to now",
    "everything from two days ago (midnight to midnight)",
    "the last 72 hours",
    "local time: 09:00 to 17:00 on June 2",
    "when daylight savings changed",
    "1 AM to 2 AM last night",
    "full day for March 15",
    "Feb 1, 08:00 until Feb 1, 20:00",
    "from 11 PM yesterday to 2 AM today",
    "the entire first week of June",
    "the last 30 minutes",
    "5 minutes before and after 10:03 AM local time",
    "Feb 14 at 8 AM to Feb 15 at 8 AM",
    "Jan 3rd from midday to end of day local",
    "the logs from 10:30 to 11:45 on June 10",
    "past 12 hours",
    "this morning 6 AM to noon",
    "all of April 10, 2024",
    "yesterday evening until this morning",
    "from the 12th to the 10th",
    "Monday 11:00 AM local time to Wednesday 11:00 AM UTC",
    "around noon Tokyo time",
    "last weekend, from Friday night to Sunday evening",
    "yesterday afternoon PT to 9 AM ET today",
    "2:34 AM EST on August 11",
    "the last big spike until now",
    "logs from midnight on Jan 1 to midnight on Jan 2",
    "sometime around last Tuesday afternoon to Wednesday morning",
    "all data from when the test environment was deployed until now",
    "the logs from 8 AM to 1 PM yesterday local time",
    "New Year's weekend",
    "from lunch time last Tuesday in Tokyo",
    "the 30 minutes after the last 502 error",
    "the day we rolled back the code in PST",
    "last 2 hours PST",
    "2 PM to 4 PM yesterday local time",
    "midnight to now",
    "when memory spiked 2 days ago",
    "the day before yesterday, 2 PM until 7 PM",
    "the day we had that major incident last month",
    "Thursday to Friday morning",
]


async def main():
    max_concurrency = 3
    semaphore = asyncio.Semaphore(max_concurrency)

    async def task(client: httpx.AsyncClient, prompt: str):
        async with semaphore:
            print(f"Request: {prompt}")
            response = await client.get(
                "http://localhost:8099/infer-time-range", params={"prompt": prompt}
            )
            print(f"Response: {response.status_code} {response.content}")

    async with httpx.AsyncClient(timeout=30.0) as client:
        async with asyncio.TaskGroup() as tg:
            while True:
                prompt = random.choice(mock_user_prompts)
                await asyncio.sleep(random.random() * 5.0)
                tg.create_task(task(client, prompt))


if __name__ == "__main__":
    asyncio.run(main())
