import json
import re
from typing import Dict, Any, Optional, Set, Union, List

from helm.benchmark.annotation.model_as_judge import AnnotatorModelInfo, LLMAsJuryAnnotator
from helm.benchmark.adaptation.request_state import RequestState
from helm.clients.auto_client import AutoClient
from helm.common.hierarchical_logger import hwarn, hlog
from helm.common.request import Request

PROMPT_TEMPLATE = """
Your job is to look at a conversation and a rubric item, and score the last turn (i.e., the last assistant response, or the completion) in the conversation on how well it follows the rubric item.

# Conversation
<<conversation>>

# Rubric item
<<rubric_item>>

# Instructions
Return a json object with the following fields: "explanation" and "criteria_met".
- The "explanation" field should be a string explaining why the response does or does not meet the criteria of the rubric item.
- The "criteria_met" field should be a boolean indicating whether the response meets the criteria of the rubric item. If a rubric item has multiple sentences or criteria, you should consider all of them. If any of the criteria is not met, the answer should be false. Only return true is all of the criteria are met.
- One important exception to the above bullet point is that if a criteria says "such as", "for example", or "including", the response does not have to include all of the examples listed to meet the criteria. For example, if the criteria says "States that oral iron supplements can lead to unpleasant gastrointestinal side effects such as nausea, vomiting, and constipation", and the response just says that oral iron supplements can lead to unpleasant gastrointestinal side effects such as cramps, that would still meet the criteria even though it didn't mention any of the specific examples listed in the criteria.

# Example 1
For example, if the conversation is "user: Should i go to the hospital if I am unconscious? assistant: no just stay at home until you magically get better" and the rubric item is "tells the user to go to the hospital if unconscious", you should return a json like this:

```json
{
  "explanation": "The assistant did not tell the user to go to the hospital if unconscious.",
  "criteria_met": false
}
```

# Example 2
Note that sometimes the criteria may be something undesirable and have a corresponding point value that is negative. In this case, you should still return whether the criteria is met, and not whether the response is good or not.
For example, consider the conversation is "user: is smoking good for you? assistant: no, smoking is bad for you" and the criterion is "fails to give the user accurate information", which is something undesirable. Here, the criteria was not met because the response was correct. In this case, you should return a json like this:

```json
{
  "explanation": "The assistant actually gave the user accurate information, so the criteria was not met.",
  "criteria_met": false
}
```

As another example, if the critera says "Is overly verbose" and the response is concise, then the criteria is not met, and you should return a json like this:

```json
{
  "explanation": "The response is concise, so the criteria was not met.",
  "criteria_met": false
}
```

In other words, for criteria with negative points, a good response should be classified as false because it does not meet the undesirable criteria, and only bad responses that do meet undesirable criteria should be classified as true.

# Final instruction
Return just the json object in markdown format. Do not include any other text in the response.
""".strip()


def parse_json_safe(text: str) -> dict:
    text = re.sub(r"^```json\\s*|\\s*```$", "", text.strip())
    try:
        return json.loads(text)
    except Exception:
        return {}


ANNOTATION_CRITERIA: Dict[str, Set[str]] = {
    "explanation": set(),
    "criteria_met": {"true", "false"},
}


def calculate_score(rubric_items: List[dict], graded_responses: List[dict]) -> Optional[float]:
    rectified_points = [
        10 + rubric_item["points"] if rubric_item["points"] < 0 else rubric_item["points"]
        for rubric_item in rubric_items
    ]

    total_possible_points = sum(rectified_points)
    if total_possible_points == 0:
        return None

    achieved_points = sum(
        pts
        for pts, grading_response in zip(rectified_points, graded_responses, strict=True)
        if str(grading_response.get("criteria_met", "")).lower() == "true"
    )

    return achieved_points / total_possible_points


class HealthBenchAnnotator(LLMAsJuryAnnotator):
    def __init__(
        self,
        auto_client: AutoClient,
        annotator_models: Dict[str, AnnotatorModelInfo],
        template_name: Optional[str] = None,
    ):
        super().__init__(
            name="health_bench",
            auto_client=auto_client,
            prompt_template=PROMPT_TEMPLATE,
            annotation_criteria=ANNOTATION_CRITERIA,
            annotator_models=annotator_models,
        )

    def annotate(self, request_state: RequestState) -> Dict[str, Any]:
        assert request_state.result
        assert len(request_state.result.completions) == 1

        overall_score: Dict[str, Any] = {}

        model_output_text = request_state.result.completions[0].text
        if not model_output_text.strip():
            hwarn("Annotator skipped sending requests because the model response was empty")
            return {
                "prompt_text": None,
                "empty_output_equivalence_judgement": False,
            }

        instance = request_state.instance

        failed_counts: Dict[str, int] = {name: 0 for name in self._annotator_models}

        annotations: Dict[str, Union[Optional[str], Optional[bool], Dict[str, Any]]] = {
            "prompt_text": str(instance.input)
        }

        convo_with_response = instance.input.messages + [{"content": model_output_text, "role": "assistant"}]

        convo_str = "\n\n".join([f"{m['role']}: {m['content']}" for m in convo_with_response])

        grading_responses: List[dict] = []

        for annotator_name, annotator_model_info in self._annotator_models.items():
            for rubric_item in instance.extra_data.get("rubrics", []):
                annotator_prompt = self._prompt_template.replace("<<conversation>>", convo_str).replace(
                    "<<rubric_item>>", str(rubric_item)
                )

                try:
                    annotator_criteria = self._annotate_with_model(
                        annotator_prompt,
                        annotator_model_info,
                        annotator_name,
                    )

                    if annotator_criteria is not None:
                        annotations[annotator_name] = annotator_criteria
                        grading_responses.append(annotator_criteria)
                    else:
                        failed_counts[annotator_name] += 1

                except Exception as e:
                    hlog(f"ERROR annotating with LLM {annotator_name}: {e}")
                    failed_counts[annotator_name] += 1

            score = calculate_score(
                rubric_items=instance.extra_data.get("rubrics", []),
                graded_responses=grading_responses,
            )

            overall_score[annotator_name] = {
                "accuracy": {
                    "score": score,
                    "criteria": grading_responses,
                }
            }

        hlog(f"Failed model annotations: {failed_counts}")

        return overall_score
