from collections import defaultdict

from .models import (
    NEVER_REHYDRATE,
    REHYDRATE_DEFAULT,
    RehydrationEvent,
    RehydrationResult,
    Replacement,
)


def rehydrate(
    cloud_response: str,
    replacements: list[Replacement],
    enabled_categories: set[str] | None = None,
) -> RehydrationResult:
    """Apply the rehydration policy to a cloud response.

    Defaults:
      - Substitute replacement -> original only for categories in REHYDRATE_DEFAULT
      - Never substitute for categories in NEVER_REHYDRATE
      - Fail closed when multiple originals share a replacement (ambiguous)
      - Fail closed when the replacement string does not appear in the response

    v0.1 uses plain substring substitution. Word-boundary handling and quoted-block
    skipping (per the rehydration policy doc) are deferred.
    """
    if enabled_categories is None:
        enabled_categories = REHYDRATE_DEFAULT

    by_replacement: dict[str, list[Replacement]] = defaultdict(list)
    for rep in replacements:
        if rep.replacement:
            by_replacement[rep.replacement].append(rep)

    text = cloud_response
    events: list[RehydrationEvent] = []

    for rep in replacements:
        if rep.category in NEVER_REHYDRATE or rep.category not in enabled_categories:
            events.append(
                RehydrationEvent(rep.original, rep.replacement, rep.category, "category_disabled")
            )
            continue
        if not rep.replacement:
            events.append(
                RehydrationEvent(rep.original, rep.replacement, rep.category, "no_replacement")
            )
            continue
        if len(by_replacement[rep.replacement]) > 1:
            events.append(
                RehydrationEvent(rep.original, rep.replacement, rep.category, "ambiguous_origin")
            )
            continue
        if rep.replacement not in text:
            events.append(
                RehydrationEvent(rep.original, rep.replacement, rep.category, "not_found")
            )
            continue
        text = text.replace(rep.replacement, rep.original)
        events.append(
            RehydrationEvent(rep.original, rep.replacement, rep.category, "substituted")
        )

    return RehydrationResult(rehydrated_response=text, events=events)
