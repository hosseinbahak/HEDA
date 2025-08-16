# app/services/consensus.py
def calculate_consensus(prosecutor_content: dict, defense_content: dict, reflection_content: dict) -> float:
    """Toy consensus: fraction of prosecutor charges that defense 'low-confidence' rebuts,
    weighted by reflection confidence."""
    case = prosecutor_content.get("case_file", [])
    rebuttals = (defense_content or {}).get("rebuttals", {})
    if not case:
        return 1.0
    weak = 0
    for charge in case:
        cid = charge.get("charge_id")
        conf = (rebuttals.get(cid) or {}).get("confidence", 1.0)
        if conf < 0.5:
            weak += 1
    base = weak / max(len(case), 1)
    qual = (reflection_content or {}).get("confidence_in_debate_quality", 0.6)
    return float(base * qual)
