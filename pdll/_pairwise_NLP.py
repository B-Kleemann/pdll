import os

import openai
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import mean_squared_error

import pdll._pairwise_NLP_dataprocessing as data_processing
import pdll._pairwise_NLP_rubricextraction as rubric_extraction

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = api_key

SEED = 17
FOLD_ID = 0

# read rubric files
essay_set_ID_to_scoring_rubrics = rubric_extraction.get_rubric_texts_from_files()


def get_pair_diff_as_int(text1: str, text2: str, rubric: str) -> int:
    prompt = f"""
    Task:
    Strictly evaluate two texts according to the rubric below.
    Rules:
    Return only one signed integer: the score difference (Text 1 score minus Text 2 score).
    Do NOT include any explanations, comments, extra characters, whitespace, or anything besides a single signed integer.
    Any output other than a single signed integer will be considered invalid.

    Rubric:
    {rubric}

    Text 1:
    {text1}
    Text 2:
    {text2}"""

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

        result = response.choices[0].message.content.strip()
        return int(result)

    except (ValueError, IndexError, AttributeError) as parse_err:
        raise RuntimeError(f"Failed to parse score difference: {parse_err}")

    except Exception as api_err:
        raise RuntimeError(f"OpenAI API call failed: {api_err}")


{
    # diff_1 = get_pair_diff_as_int(
    #     X_train.iloc[0].essay, X_train.iloc[1].essay, rubric_set_1_text
    # )
    # diff_2 = get_pair_diff_as_int(
    #     X_train.iloc[0].essay, X_train.iloc[1].essay, rubric_set_1_text
    # )
    # if diff_1 != diff_2:
    #     print(f"Different results! {diff_1} != {diff_2}")
    # else:
    #     print(f"Same results! Both say {diff_1}")
    # ? variation of the differences is high, so the model is not consistent in its predictions
    # ? range tested with this essay pair is -1 to 3, -1 x 1, 0 x 3, 1 x 2, 2 x 3, 3 x 1
}  # type: ignore


def predict_scores(test_data: pd.DataFrame, training_data: pd.DataFrame, rubric: str):

    predictions = []
    # renaming for clarity: data = test_data, baseline = training_data

    for i, row_i in test_data.iterrows():
        store_pred_scores = []

        for j, row_j in training_data.iterrows():
            # if i <= j:  # type: ignore
            # try <= and >= both, this halfs the squared data, do that as double checking, does full set work better or is half-set sufficient already in obtaining good results,
            # Todo for debugging: isolated predictions, give same pair twice, then see if difference is 0
            # continue
            try:
                diff = get_pair_diff_as_int(row_i["essay"], row_j["essay"], rubric)
                score_of_baseline_essay = row_j.iloc[1]
                score_pred = score_of_baseline_essay + diff
                store_pred_scores.append(score_pred)
            except RuntimeError:
                continue

        avg_score = 0
        # if the list is empty, it means no valid differences were found
        if len(store_pred_scores) != 0:
            avg_score = sum(store_pred_scores) / len(store_pred_scores)

        predictions.append(round(avg_score, 2))

    test_data["y_pred"] = predictions
    return test_data


essay_set_ID = 1  # Set the prompt ID you want to use
as_list_of_tuples = True
data_train, data_dev, data_test = data_processing.get_data(
    FOLD_ID, essay_set_ID, as_list_of_tuples
)

limit = 6  # Limit the number of rows for testing
limit_data = limit  # Limit the number of rows for testing
limit_baseline = limit  # Limit the number of rows for testing

assert limit > 0, "Limit must be greater than 0."
assert limit < len(data_train), "Limit exceeds number of rows in dataset."
assert limit <= 6, "Limit exceeds number of reasonable rows."

# conversion to DataFrame
data_train, data_dev, data_test = (
    pd.DataFrame(data_train, columns=["essay", "score"]),
    pd.DataFrame(data_dev, columns=["essay", "score"]),
    pd.DataFrame(data_test, columns=["essay", "score"]),
)

#! limits data for development and testing
data_train, data_dev, data_test = (
    # data_train.tail(limit_baseline),
    # data_dev.tail(limit_data),
    # data_test.tail(limit),
    data_train.head(limit_baseline),
    data_dev.head(limit_data),
    data_test.head(limit),
)

score_prediction = predict_scores(
    pd.DataFrame(data_dev),
    pd.DataFrame(data_train),
    essay_set_ID_to_scoring_rubrics[essay_set_ID],
)
print(score_prediction)

y_true = score_prediction["score"]
y_pred = score_prediction["y_pred"]
mse = mean_squared_error(y_true, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# * DONE
# include support for the other essay sets (other rubrics)
# connect my work to the pre-folded / split data instead of the test-version

# make sure the model doesn't have access to the internet so it doesn't just look-up;
# api calls do not have native access to the internet

# * TODO
#! baseline first!
# todo: with increasing complexity, bug detection might be more difficult
# todo: include separation between modifications that should and shouldn't affect the score (for example cashing = no impact on score, prompt change = impact on score)
# todo: implement a baseline predictor that only predicts the score directly from the model, for comparison

# todo: cash get_pair_diff_as_int , reduces the API calls, use the FULL string, not only the inputs, because for example the prompt could be modified from one call to the other, cashing as dataframe, string and int diff for querying, printing to see which is a fresh API call and which come form the cash


# ! thesis will get registered now, this means an official DEADLINE, I will get an e-mail about that
# new title: Pairwise Difference Learning for LLMs
