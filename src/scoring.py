from __future__ import annotations

from dataclasses import dataclass


ACTIONS = [
    "Continue pumping",
    "Reduce PPA / prepare flush",
    "Hold PPA",
    "Increase PPA",
    "Decrease rate",
    "Increase rate",
    "Flush",
    "Shut down",
    "Check pumps / stabilize rate",
    "Verify sensor / compare channels",
    "Evaluate offset communication",
]


@dataclass(frozen=True)
class DecisionResult:
    score: int
    feedback: str


@dataclass(frozen=True)
class ScenarioInfo:
    expected_diagnosis: str
    recommended_action: str
    training_note: str


DIAGNOSIS_OPTIONS = [
    "Normal treatment response",
    "Possible screenout",
    "Possible perforation or cluster restriction",
    "Pump or rate-control issue",
    "Possible frac hit / pressure communication",
    "Possible sensor or data quality issue",
]


ACTION_OPTIONS = [
    "Continue pumping",
    "Reduce PPA / prepare flush",
    "Hold PPA",
    "Increase PPA",
    "Decrease rate",
    "Increase rate",
    "Flush",
    "Shut down",
    "Check pumps / stabilize rate",
    "Verify sensor / compare channels",
    "Evaluate offset communication",
]


SCENARIO_INFO: dict[str, ScenarioInfo] = {
    "Normal Job": ScenarioInfo(
        expected_diagnosis="Normal treatment response",
        recommended_action="Continue pumping",
        training_note=(
            "Smooth pressure response. Rate is stable, PPA follows schedule, "
            "and friction increases gradually as slurry gets heavier."
        ),
    ),
    "Screenout": ScenarioInfo(
        expected_diagnosis="Possible screenout",
        recommended_action="Reduce PPA / prepare flush",
        training_note=(
            "At stable rate and increasing proppant concentration, treating pressure and "
            "net pressure climb quickly. The fracture is losing acceptance or proppant is bridging."
        ),
    ),
    "Perforation Plugging": ScenarioInfo(
        expected_diagnosis="Possible perforation or cluster restriction",
        recommended_action="Hold PPA",
        training_note=(
            "Surface pressure rises mainly because perforation friction rises. Estimated "
            "BHP and net pressure may not rise as strongly as surface treating pressure."
        ),
    ),
    "Pump Issue": ScenarioInfo(
        expected_diagnosis="Pump or rate-control issue",
        recommended_action="Check pumps / stabilize rate",
        training_note=(
            "Rate becomes unstable or drops. Pressure response follows the rate disturbance. "
            "Do not confuse this with formation-driven pressure growth."
        ),
    ),
    "Frac Hit": ScenarioInfo(
        expected_diagnosis="Possible frac hit / pressure communication",
        recommended_action="Evaluate offset communication",
        training_note=(
            "At roughly stable pumping conditions, pressure suddenly drops or changes slope. "
            "This can represent communication with an offset well or a new pressure sink."
        ),
    ),
    "Sensor Error": ScenarioInfo(
        expected_diagnosis="Possible sensor or data quality issue",
        recommended_action="Verify sensor / compare channels",
        training_note=(
            "Measured surface pressure shows unrealistic spikes or jumps that are not "
            "supported by rate, PPA, friction, or BHP behavior."
        ),
    ),
}


DECISION_TABLE: dict[str, dict[str, DecisionResult]] = {
    "Normal Job": {
        "Maintain": DecisionResult(10, "Good call. The job is stable."),
        "Reduce PPA": DecisionResult(0, "Safe, but it gives up placement with no clear warning."),
        "Increase Rate": DecisionResult(-5, "Unneeded rate increase adds pressure risk."),
        "Flush": DecisionResult(-8, "Premature flush would end the stage early."),
        "Drop Rate": DecisionResult(-2, "Safe but not required for the current response."),
        "Check Sensor": DecisionResult(2, "Always reasonable, but not the main decision here."),
    },
    "Screenout": {
        "Maintain": DecisionResult(-12, "Dangerous. Pressure and net pressure are building."),
        "Reduce PPA": DecisionResult(10, "Correct. Reducing proppant loading lowers screenout risk."),
        "Increase Rate": DecisionResult(-18, "Dangerous. More rate can worsen the pressure climb."),
        "Flush": DecisionResult(15, "Strong response. Flush is appropriate for a late screenout trend."),
        "Drop Rate": DecisionResult(6, "Safe action, but proppant concentration also needs attention."),
        "Check Sensor": DecisionResult(0, "Verification helps, but the hydraulic trend is real."),
    },
    "Perforation Plugging": {
        "Maintain": DecisionResult(-8, "Risky. Perf friction is moving against you."),
        "Reduce PPA": DecisionResult(8, "Good response. Lower solids can reduce plugging tendency."),
        "Increase Rate": DecisionResult(-12, "Risky. Higher rate increases perf friction."),
        "Flush": DecisionResult(10, "Good defensive action if restriction continues."),
        "Drop Rate": DecisionResult(7, "Safe. Lower rate reduces pressure across the perforations."),
        "Check Sensor": DecisionResult(1, "Useful check, but friction behavior is the main signal."),
    },
    "Pump Issue": {
        "Maintain": DecisionResult(-6, "Rate loss needs a surface-equipment response."),
        "Reduce PPA": DecisionResult(4, "Conservative and usually safe while equipment is unstable."),
        "Increase Rate": DecisionResult(-10, "Not the first move until the pump issue is understood."),
        "Flush": DecisionResult(2, "May be safe, but first diagnose the rate loss."),
        "Drop Rate": DecisionResult(10, "Good. Stabilize equipment before pushing slurry."),
        "Check Sensor": DecisionResult(7, "Good. Confirm instrumentation and pump-side readings."),
    },
    "Frac Hit": {
        "Maintain": DecisionResult(-6, "Pressure relief can hide communication risk."),
        "Reduce PPA": DecisionResult(6, "Good conservative move while communication is assessed."),
        "Increase Rate": DecisionResult(-14, "Dangerous. More rate can intensify the hit."),
        "Flush": DecisionResult(8, "Safe defensive action if offset-well communication is suspected."),
        "Drop Rate": DecisionResult(10, "Good. Reducing rate limits communication intensity."),
        "Check Sensor": DecisionResult(2, "Useful, but the pressure break needs an operational response."),
    },
    "Sensor Error": {
        "Maintain": DecisionResult(2, "Reasonable if other channels remain stable."),
        "Reduce PPA": DecisionResult(-2, "Premature. Confirm the measurement first."),
        "Increase Rate": DecisionResult(-6, "Avoid rate changes until the signal is verified."),
        "Flush": DecisionResult(-8, "Too aggressive for an instrument issue."),
        "Drop Rate": DecisionResult(0, "Safe, but it may not address the real problem."),
        "Check Sensor": DecisionResult(10, "Correct. Verify the pressure channel before changing the job."),
    },
}


def evaluate_decision(scenario: str, action: str) -> DecisionResult:
    scenario_table = DECISION_TABLE.get(scenario)
    if scenario_table is None:
        return DecisionResult(0, "Scenario is not configured.")
    return scenario_table.get(action, DecisionResult(0, "Action is not configured."))


def score_answer(scenario: str, diagnosis: str, action: str) -> DecisionResult:
    info = SCENARIO_INFO.get(scenario)
    if info is None:
        return DecisionResult(0, "Scenario is not configured.")

    score = 0
    feedback_parts: list[str] = []

    if diagnosis == info.expected_diagnosis:
        score += 10
        feedback_parts.append("Diagnosis correct.")
    else:
        score -= 5
        feedback_parts.append(f"Diagnosis mismatch. Expected: {info.expected_diagnosis}.")

    if action == info.recommended_action:
        score += 5
        feedback_parts.append("Action aligned with safe response.")
    else:
        score -= 3
        feedback_parts.append(f"Action mismatch. Recommended: {info.recommended_action}.")

    return DecisionResult(score, " ".join(feedback_parts))
