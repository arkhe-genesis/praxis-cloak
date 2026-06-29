"""Stage 5 of the decomposed pipeline: parse the rewriter output.

The rewriter emits the protected message as plain prose, with each
generalization wrapped in `<gN>...</gN>` tags where N is the substitution
number from the user-message prompt. We:

  1. Extract every `<gN>...</gN>` keyed by N.
  2. Strip all `<gN>` / `</gN>` tags to produce the clean protected message.
  3. Build the replacement map by:
     - taking pseudonym swaps from what we ourselves chose in Stage 3,
     - looking up the rewriter's tag content for each generalize span by its
       substitution number (not by left-to-right position — that was the v0.1
       parser bug that misaligned every span downstream of a dropped one).
"""

import re

from .models import Replacement

NUMBERED_TAG_PATTERN = re.compile(r"<g(\d+)>(.*?)</g\1>", re.DOTALL)
STRIP_TAG_PATTERN = re.compile(r"</?g\d+>")


def parse_rewriter_output(
    raw_output: str,
    name_subs: dict[str, str],
    generalize_spans: list[dict],
    generalize_numbers: list[int],
) -> tuple[str, list[Replacement]]:
    """Return (clean_protected_message, replacements).

    `generalize_numbers[i]` is the substitution number assigned to
    `generalize_spans[i]` in the user message handed to the rewriter.
    """
    content_by_number: dict[int, str] = {}
    for n_str, content in NUMBERED_TAG_PATTERN.findall(raw_output):
        try:
            content_by_number[int(n_str)] = content
        except ValueError:
            continue

    clean_message = STRIP_TAG_PATTERN.sub("", raw_output).strip()

    replacements: list[Replacement] = []
    for orig, pseudo in name_subs.items():
        replacements.append(Replacement(original=orig, replacement=pseudo, category="name"))

    for span, n in zip(generalize_spans, generalize_numbers):
        replacement = content_by_number.get(n, "")
        replacements.append(
            Replacement(
                original=span["span"],
                replacement=replacement,
                category=span["category"],
            )
        )

    return clean_message, replacements
