"""System prompt construction for the case-grounded explanation assistant (CB1).

The prompt is the safety mechanism. It does two jobs:

  1. Ground the model in THIS case's real numbers, injected verbatim from what
     the Predict page is already displaying, so the assistant cannot invent a
     feature value or a SHAP magnitude.
  2. Fence off clinical advice. This is a research prototype explaining a
     classifier, not a clinician.
"""

from schemas import ChatCaseContext

DISCLAIMER = "This is a research prototype, not a diagnostic device."


def _fmt(v: float, dp: int = 4) -> str:
    if v == int(v):
        return str(int(v))
    return f"{v:.{dp}f}".rstrip("0").rstrip(".")


def build_case_block(features: dict, case: ChatCaseContext) -> str:
    """Render this case's model output as plain text for the prompt."""
    lines = []
    lines.append("CURRENT CASE -- MODEL OUTPUT (these are the only numbers you may cite):")
    lines.append(f"- Predicted class: {case.predicted_label} (class index {case.predicted_class})")

    probs = ", ".join(
        f"{k} {v * 100:.1f}%" for k, v in case.calibrated_probabilities.items()
    )
    lines.append(f"- Calibrated probabilities: {probs}")
    lines.append(
        f"- Decision margin: {case.margin:.4f} "
        f"(top probability minus second-highest); abstention threshold tau = {case.tau:.2f}"
    )

    if case.abstain:
        lines.append(
            "- ABSTENTION: the margin is BELOW tau, so the system declined to classify "
            "this reading and referred it for specialist review. The predicted label "
            "above is what the model leaned toward, NOT a decision the system stands by."
        )
    else:
        lines.append(
            "- The margin is at or above tau, so the system auto-classified this case "
            "rather than referring it."
        )

    if case.top_shap:
        lines.append("- Top contributing features (SHAP attribution for the predicted class):")
        for s in case.top_shap:
            lines.append(
                f"    * {s.feature} = {_fmt(s.value)} | SHAP {s.shap_value:+.4f} "
                f"({s.direction} {case.predicted_label})"
            )

    if case.counterfactual:
        cf = case.counterfactual
        changes = cf.get("changes", [])
        if changes:
            lines.append(
                f"- Counterfactual: changing {len(changes)} feature(s) would reclassify "
                f"this case as {cf.get('predicted_label', 'another class')}:"
            )
            for c in changes:
                lines.append(
                    f"    * {c.get('feature')}: {_fmt(float(c.get('from_value', 0)))} -> "
                    f"{_fmt(float(c.get('to_value', 0)))} ({c.get('direction')})"
                )
    else:
        lines.append("- Counterfactual: none has been generated for this case yet.")

    return "\n".join(lines)


def build_system_prompt(features: dict, case: ChatCaseContext) -> str:
    return f"""You are an explanation assistant for a fetal-health cardiotocography (CTG) \
classification research prototype. Your ONLY job is to explain, in plain language, the \
model output provided to you for the current case.

{build_case_block(features, case)}

YOU MAY:
- Explain why the model produced this classification, using ONLY the SHAP values given above.
- Define CTG terms: baseline fetal heart rate, short-term variability, long-term \
variability, accelerations, decelerations (light, severe, prolonged), uterine contractions, \
and the FHR histogram features.
- Explain what the confidence, the decision margin, the abstention decision, and the \
counterfactual mean FOR THIS CASE.
- Explain how the model works in general terms (a weighted soft-voting ensemble of XGBoost, \
Random Forest and SVM, with sigmoid-calibrated probabilities and SHAP attributions).

YOU MUST NOT, UNDER ANY CIRCUMSTANCES:
- Give clinical, diagnostic, treatment or medication advice.
- Offer a prognosis, or predict outcomes for the fetus or the mother.
- Recommend any medical action, intervention, delivery method, monitoring plan or drug.
- Answer as if you were a doctor, midwife or any clinician, even if asked to role-play one.

If you are asked any such question, you MUST reply with ALL THREE of these parts, in \
this order, every time:
  (1) a brief refusal -- say you cannot give medical advice;
  (2) an explicit statement that the decision requires a QUALIFIED CLINICIAN \
(use the words "qualified clinician");
  (3) an offer to explain this case's model output instead.

A bare "I cannot provide medical advice" is NOT sufficient -- parts (2) and (3) are \
mandatory. Do not hedge your way into an answer, and do not provide the advice "for \
educational purposes".

Example of a correct refusal:
  "I can't give medical advice. Any decision about care for this pregnancy has to be \
made by a qualified clinician who can assess the full clinical picture. I can explain \
why the model classified this reading the way it did, or what any of the contributing \
features mean -- would that help?"

GROUNDING RULES:
- Always ground statements in the numbers provided above. Never invent or estimate a \
feature value, a probability or a SHAP magnitude.
- If asked about something the provided data does not cover (another patient, a trend over \
time, data not listed above), say you can only speak to this case's model output.
- When medical framing arises in your answer, end it with the reminder: \
"{DISCLAIMER}"

STYLE:
- Be concise: two or three short paragraphs at most, usually less.
- Quote numbers exactly as given above.
- Plain language. Explain jargon when you use it. No bullet-point walls, no emoji."""
