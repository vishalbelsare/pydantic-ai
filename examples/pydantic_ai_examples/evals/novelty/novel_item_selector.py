# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "openai>=1.14.0",
#   "numpy>=1.26",
# ]
# ///

"""novel_selector.py

A self-contained helper for incrementally sampling the most *novel-yet-prevalent*
items from a collection (e.g., user prompts recorded in an observability system),
suitable for building evaluation sets.

Run with:  uv run novel_selector.py
"""

from __future__ import annotations

import os
import sys
from collections import Counter, deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Deque, List, Any, Generic, Callable

import numpy as np
from openai import OpenAI
from typing_extensions import TypeVar


@dataclass(slots=True)
class SelectorConfig:
    """
    Configuration for the `NovelItemSelector`.
    """
    model: str = 'text-embedding-3-small'
    alpha: float = 0.6  # prevalence weight in [0, 1]
    distance_threshold: float = 0.3  # cosine distance below which an item is “covered”
    batch_size: int = 512  # for embedding API calls


def _l2_normalise(arr: np.ndarray[Any, Any]) -> None:  # in-place
    arr /= np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12


def _batched(iterable: list[str], n: int) -> Iterable[list[str]]:
    for i in range(0, len(iterable), n):
        yield iterable[i : i + n]


T = TypeVar('T', default=str)
class NovelItemSelector(Generic[T]):
    """Incrementally surfaces the next unhandled item with the highest prevalence-weighted novelty.

    This class is designed to be used in scenarios where you have a large set of
    unhandled items (e.g., prompts, queries) and an (optional) smaller set of already handled
    items. It uses embeddings to compute novelty based on cosine distance to the handled items,
    but also considers the prevalence of each item in the unhandled set.

    As items are yielded, they are moved from the unhandled to the handled set.
    """

    def __init__(
        self,
        unhandled_inputs: Sequence[T],
        handled_inputs: Sequence[T] | None = None,
        render_inputs: Callable[[T], str] = str,
        *,
        config: SelectorConfig | None = None,
        client: OpenAI | None = None,
    ) -> None:
        # TODO: This class needs to be refactored to do less in the constructor.
        handled_inputs = handled_inputs or []

        self.cfg = config or SelectorConfig()
        self.embedding_client = client or OpenAI()
        self.render_inputs = render_inputs

        # ---- dedupe + count ----
        unhandled_strings = [render_inputs(item) for item in unhandled_inputs]
        self._counts: Counter[str] = Counter(unhandled_strings)
        self._unique_unhandled_strings: List[str] = list(self._counts)
        
        # Create mapping from string back to original object
        string_to_obj: dict[str, T] = {}
        for item in unhandled_inputs:
            string_to_obj[render_inputs(item)] = item
        self._unique_unhandled: List[T] = [string_to_obj[s] for s in self._unique_unhandled_strings]
        
        self._n = len(self._unique_unhandled)
        self._picked: Deque[int] = deque()  # indices already returned

        # ---- embed everything ----
        self._unhandled_vecs = self._embed(self._unique_unhandled_strings)
        handled_strings = [render_inputs(item) for item in set(handled_inputs)] if handled_inputs else []
        self._handled_vecs = (
            self._embed(handled_strings)
            if handled_strings
            else np.empty((0, 0))
        )

        # ---- compute initial novelty (distance to nearest handled) ----
        if self._handled_vecs.size:
            sim = self._unhandled_vecs @ self._handled_vecs.T  # (N, L)
            self._d = 1.0 - sim.max(axis=1)  # cosine *distance*
        else:
            self._d = np.ones(self._n, dtype=np.float32)

    # ----------------------- public API ----------------------- #

    def pop_next(self) -> tuple[T, float] | None:
        """Return *and mark* the most novel-yet-prevalent prompt
        that is still beyond the coverage threshold.

        Returns None when everything remaining is already covered.
        """
        # utility = novelty * prevalence
        novelty = self._d
        weight = (
            np.log1p([self._counts[t] for t in self._unique_unhandled_strings])
            ** self.cfg.alpha
        )
        utility = novelty * weight

        # mask out already picked items
        for idx in self._picked:
            utility[idx] = -1.0

        idx = int(utility.argmax())
        item_utility = utility[idx]
        if item_utility <= self.cfg.distance_threshold:
            return None  # all done 🎉

        self._picked.append(idx)
        self._update_distances(idx)
        return self._unique_unhandled[idx], item_utility

    def _embed(self, texts: List[str]) -> np.ndarray[Any, Any]:
        out: List[np.ndarray[Any, Any]] = []
        for chunk in _batched(texts, self.cfg.batch_size):
            rsp = self.embedding_client.embeddings.create(input=chunk, model=self.cfg.model)
            vecs = np.array([item.embedding for item in rsp.data], dtype=np.float32)
            out.append(vecs)
        arr = np.vstack(out)
        _l2_normalise(arr)
        return arr

    def _update_distances(self, new_idx: int) -> None:
        """One dot-product updates the whole distance cache O(N·d)."""
        center = self._unhandled_vecs[new_idx : new_idx + 1]  # shape (1,d)
        sim = self._unhandled_vecs @ center.T  # (N,1)
        new_d = 1.0 - sim.ravel()
        self._d = np.minimum(self._d, new_d)


# ----------------------------- demo usage ----------------------------- #

def main():
    if 'OPENAI_API_KEY' not in os.environ:
        sys.exit('❌  Please set OPENAI_API_KEY in your environment.')

    # 8 clusters, intentionally imbalanced
    unhandled = (
        # GREETINGS  (very common, 20)
        ["hi there!", "hello!", "good morning!", "hey!"] * 5
        # WEATHER  (common, 15)
        + [
            "what's the weather in Paris?",
            "weather Seattle tomorrow",
            "umbrella today?",
            "temperature in Tokyo",
            "wind speed in Chicago",
        ]
        * 3
        # PYTHON ERRORS (medium, 10)
        + [
            "TypeError: 'NoneType' object is not subscriptable",
            "IndexError list assignment index out of range",
        ]
        * 5
        # SQL TUNING (medium-rare, 6)
        + [
            "how to speed up Postgres join?",
            "why is my query slow postgres",
            "create index concurrently example",
        ]
        * 2
        # TRANSLATE  (rare, 4)
        + [
            "translate 'good night' to German",
            "翻译 'machine learning' 成中文",
        ]
        * 2
        # QUANTUM JOKE  (ultra-rare, 1)
        + [
            "tell me a knock-knock joke about quantum physics"
        ]
        # FOOD SUBSTITUTION (rare, 4)
        + [
            "substitute buttermilk in pancakes",
            "替代牛奶烘焙",
        ]
        * 2
        # ESA MEMORY LIMIT (rare, 3)
        + [
            "why does my embedded system crash at 0x20000000?",
            "cortex-m stack overflow detection",
            "how to tune linker script for SRAM",
        ]
    )
    handled = list[str]()
    selector = NovelItemSelector(
        unhandled, handled, render_inputs=lambda x: x, config=SelectorConfig(distance_threshold=0.7)
    )
    for i in range(20):
        nxt = selector.pop_next()
        if nxt is None:
            break
        print(f'{i + 1:2}.', nxt[0])

if __name__ == '__main__':
    main()