# app/services/bias.py
def detect_biases(prosecutor_content: dict, defense_content: dict, text: str):
    biases = []
    # simple length heuristic
    if len(text.split()) > 300 and len(prosecutor_content.get("case_file", [])) > 5:
        biases.append("length_bias")
    # simple overconfidence heuristic
    pconfs = [c.get("confidence", 0.0) for c in prosecutor_content.get("case_file", [])]
    if pconfs and max(pconfs) > 0.98:
        biases.append("overconfidence_bias")
    return biases
