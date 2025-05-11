import os

import openai
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = api_key


def get_essay_score_as_int(essay: str, rubric: str) -> int:
    prompt = f"""
    Task:
    Strictly evaluate the essay according to the rubric below.
    
    Rules:
    Return only one integer: the score for the essay.
    Do NOT include any explanations, comments, extra characters, whitespace, or anything besides a single integer.
    Any output other than a single integer will be considered invalid.

    Rubric:
    {rubric}

    Essay:
    {essay}
    """

    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert text comparison assistant.",
                },
                {"role": "user", "content": prompt},
            ],
        )

        result = response.choices[0].message.content.strip()  # type: ignore
        return int(result)

    except (ValueError, IndexError, AttributeError) as parse_err:
        raise RuntimeError(f"Failed to parse score difference: {parse_err}")

    except Exception as api_err:
        raise RuntimeError(f"OpenAI API call failed: {api_err}")


def predict_scores_solo(test_data: pd.DataFrame, rubric: str) -> pd.DataFrame:

    predictions = []
    for i, row_i in test_data.iterrows():
        try:
            # the score here is predicted twice to mimic the composition of the original score (the sum of two single scores by two different experts)
            score_pred_1 = get_essay_score_as_int(row_i["essay"], rubric)
            score_pred_2 = get_essay_score_as_int(row_i["essay"], rubric)

            score_pred = score_pred_1 + score_pred_2
            predictions.append(score_pred)

        except RuntimeError:
            continue

    test_data["y_pred"] = predictions

    return test_data
