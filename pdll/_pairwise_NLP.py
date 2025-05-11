import os

import openai
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = api_key


def get_pair_diff_as_int(text1: str, text2: str, rubric: str) -> int:
    prompt = f"""
    Task:
    Strictly evaluate the two essays according to the rubric below.
    
    Rules:
    Return only one signed integer: the score difference (Text 1 score minus Text 2 score).
    Do NOT include any explanations, comments, extra characters, whitespace, or anything besides a single signed integer.
    Any output other than a single signed integer will be considered invalid.

    Rubric:
    {rubric}

    Text 1:
    {text1}
    
    Text 2:
    {text2}
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


# {
#     # diff_1 = get_pair_diff_as_int(
#     #     X_train.iloc[0].essay, X_train.iloc[1].essay, rubric_set_1_text
#     # )
#     # diff_2 = get_pair_diff_as_int(
#     #     X_train.iloc[0].essay, X_train.iloc[1].essay, rubric_set_1_text
#     # )
#     # if diff_1 != diff_2:
#     #     print(f"Different results! {diff_1} != {diff_2}")
#     # else:
#     #     print(f"Same results! Both say {diff_1}")
#     # ? variation of the differences is high, so the model is not consistent in its predictions
#     # ? range tested with this essay pair is -1 to 3, -1 x 1, 0 x 3, 1 x 2, 2 x 3, 3 x 1
# }  # type: ignore


def predict_scores_pairwise(
    test_data: pd.DataFrame, training_data: pd.DataFrame, rubric: str
) -> pd.DataFrame:

    predictions = []
    # renaming for clarity: data = test_data, baseline = training_data

    for i, row_i in test_data.iterrows():
        store_pred_scores = []

        for j, row_j in training_data.iterrows():
            print(f"Datapoint {i} & Datapoint {j}")

            # if i <= j:  # type: ignore
            # try <= and >= both, this halfs the squared data, do that as double checking, does full set work better or is half-set sufficient already in obtaining good results,
            # Todo for debugging: isolated predictions, give same pair twice, then see if difference is 0
            # continue
            try:
                diff = get_pair_diff_as_int(row_i["essay"], row_j["essay"], rubric)
                print(f"Diff: {diff}")

                score_of_baseline_essay = int(row_j.iloc[1])
                print(f"Score of baseline essay: {score_of_baseline_essay}")

                score_pred = score_of_baseline_essay + diff
                print(f"Score prediction: {score_pred}")

                store_pred_scores.append(score_pred)

            except RuntimeError:
                continue

        avg_score = 0
        # if the list is empty, it means no valid differences were found
        if store_pred_scores != []:
            avg_score = int(round(sum(store_pred_scores) / len(store_pred_scores), 0))
        print(f"Avg score: {avg_score}\n\n")

        predictions.append(int(avg_score))

    test_data["y_pred"] = predictions
    return test_data
