"""Stage 3 of the decomposed pipeline: deterministic pseudonym assignment.

Given a list of detected names and the original message, pick a real-name
pseudonym for each. Tries to gender-match so existing pronouns stay correct.

Pools are small and curated. Pool exhaustion within one message is unlikely;
if it happens we fall back to a suffixed neutral name.
"""

import re

FEMALE_NAMES = [
    "Mira", "Asha", "Karen", "Aisha", "Elena", "Priya", "Maya",
    "Nora", "Sofia", "Lila", "Zara", "Iris", "Naomi", "Hana",
]
MALE_NAMES = [
    "Liam", "Marcus", "Daniel", "Samir", "Mateo", "Kenji", "Ryan",
    "Owen", "Jamal", "Diego", "Theo", "Felix", "Arjun", "Noah",
]
NEUTRAL_NAMES = [
    "Alex", "Sam", "Jordan", "Taylor", "Riley", "Casey", "Avery",
    "Quinn", "Morgan", "Reese",
]

# Coarse known-gender mapping for common Western/South-Asian first names.
# Used as a first cut; falls back to pronoun heuristic.
KNOWN_GENDERS: dict[str, str] = {
    # Female-coded
    "anna": "f", "sarah": "f", "lisa": "f", "jen": "f", "jennifer": "f",
    "emma": "f", "priya": "f", "maya": "f", "ava": "f", "aisha": "f",
    "elena": "f", "sofia": "f", "olivia": "f", "rachel": "f", "karen": "f",
    "linda": "f", "mary": "f", "patricia": "f", "michelle": "f", "amy": "f",
    "lisa": "f", "amanda": "f", "stephanie": "f",
    # Male-coded
    "mark": "m", "andre": "m", "carlos": "m", "brad": "m", "ryan": "m",
    "tom": "m", "daniel": "m", "michael": "m", "david": "m", "james": "m",
    "robert": "m", "john": "m", "william": "m", "chris": "m", "matt": "m",
    "matthew": "m", "kevin": "m", "brian": "m", "scott": "m", "paul": "m",
    "jay-z": "m", "liam": "m", "mateo": "m", "marcus": "m",
    # Gender-neutral
    "sam": "n", "alex": "n", "jordan": "n", "taylor": "n", "casey": "n",
    "morgan": "n", "riley": "n", "avery": "n", "quinn": "n",
}


def infer_gender(name: str, raw_message: str) -> str:
    """Return 'f', 'm', or 'n'.

    First tries the known-name dictionary. Falls back to pronoun counts in the
    surrounding message. Last resort: 'n' (neutral).
    """
    key = name.strip().lower()
    if key in KNOWN_GENDERS:
        return KNOWN_GENDERS[key]

    text = raw_message.lower()
    female = len(re.findall(r"\b(she|her|hers|herself)\b", text))
    male = len(re.findall(r"\b(he|him|his|himself)\b", text))
    they = len(re.findall(r"\b(they|them|their|theirs|themself|themselves)\b", text))

    if female > male and female >= they:
        return "f"
    if male > female and male >= they:
        return "m"
    if they > 0 and they >= female and they >= male:
        return "n"
    return "n"


def assign_pseudonyms(
    names: list[str], raw_message: str, existing: dict[str, str] | None = None
) -> dict[str, str]:
    """Assign distinct pseudonyms to a list of names.

    Same input name gets the same pseudonym (consistent within a message).
    Different input names get different pseudonyms (entity distinctness).

    `existing` carries assignments forward across turns of a conversation, so a name
    seen earlier keeps the same pseudonym (and its pseudonym is not reused for a new
    name). The returned dict includes both the existing and any new assignments.
    """
    result: dict[str, str] = dict(existing or {})
    used: set[str] = set(result.values())

    for name in names:
        if name in result:
            continue
        gender = infer_gender(name, raw_message)
        pool = {"f": FEMALE_NAMES, "m": MALE_NAMES, "n": NEUTRAL_NAMES}[gender]
        chosen = None
        for candidate in pool:
            if candidate not in used and candidate.lower() != name.strip().lower():
                chosen = candidate
                break
        if chosen is None:
            # Pool exhausted — fall back to a numbered neutral
            chosen = f"Alex{len(used) + 1}"
        used.add(chosen)
        result[name] = chosen

    return result
