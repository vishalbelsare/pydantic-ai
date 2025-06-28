# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "openai>=1.14.0",
#   "numpy>=1.26",
# ]
# ///

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from pydantic_ai_examples.evals.novelty.novel_item_selector import NovelItemSelector


def main():
    if 'OPENAI_API_KEY' not in os.environ:
        sys.exit('❌  Please set OPENAI_API_KEY in your environment.')

    data = json.loads(
        (Path(__file__).parent / 'time_range_agent.json').read_text()
    )
    unhandled = [content for x in data if (content := x['attributes']['all_messages_events'][1]['content'])]
    handled = list[str]()
    selector = NovelItemSelector(unhandled, handled)
    for i in range(20):
        nxt = selector.pop_next()
        if nxt is None:
            break
        print(f'{i + 1:2}.', nxt[0])


if __name__ == '__main__':
    main()