import codecs
import os
from pathlib import Path
from sklearn.metrics import mean_squared_error

import openai
import pandas as pd
from dotenv import load_dotenv
from pdfquery import PDFQuery
from sklearn.model_selection import train_test_split

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = api_key

SEED = 17


# read rubric files

rubric_dir = Path(
    "C:/Users/betti/Documents/Daten-lokal/Studium/Bachelorarbeit/asap-aes/scoring_rubrics"
)

filename_rubric_set_1 = rubric_dir / "rubric_set_1.pdf"

rubric_set_1 = PDFQuery(filename_rubric_set_1)
rubric_set_1.load()

# Use CSS-like selectors to locate the elements
rubric_set_1_text = rubric_set_1.pq("LTTextLineHorizontal")

# # Extract the text from the elements
# text = [t.text for t in text_elements]

# print(text)


# Input data file from corresponding fold
FOLD_ID = 0
dir = Path(f"data-set\\asap\\fold_{FOLD_ID}")
paths = [f"{dir}\\train.tsv", f"{dir}\\dev.tsv", f"{dir}\\test.tsv"]
prompt_id_to_score_index_mapping = {
    1: 6,
    2: 3,  #! todo: complete this mapping for all prompt ids
    3: 3,
    4: 3,
    5: 3,
    6: 3,
    7: 3,
}


def read_dataset(file_path, prompt_id, score_index):
    data_x, data_y, prompt_ids = [], [], []

    with codecs.open(file_path, mode="r", encoding="UTF8") as input_file:
        next(input_file)
        for line in input_file:
            tokens = line.strip().split("\t")

            essay_id = int(tokens[0])
            essay_set = int(tokens[1])
            content = str(tokens[2].strip())
            score = float(tokens[score_index])

            if essay_set == prompt_id or prompt_id <= 0:
                data_x.append(content)
                data_y.append(score)
                prompt_ids.append(essay_set)

    return data_x, data_y, prompt_ids


def get_data(paths, prompt_id):
    train_path, dev_path, test_path = paths[0], paths[1], paths[2]

    train_x, train_y, train_prompts = read_dataset(
        train_path, prompt_id, prompt_id_to_score_index_mapping[prompt_id]
    )

    dev_x, dev_y, dev_prompts = read_dataset(
        dev_path, prompt_id, prompt_id_to_score_index_mapping[prompt_id]
    )

    test_x, test_y, test_prompts = read_dataset(
        test_path, prompt_id, prompt_id_to_score_index_mapping[prompt_id]
    )

    train = (train_x, train_y)
    dev = (dev_x, dev_y)
    test = (test_x, test_y)

    return train, dev, test


def get_pair_diff_as_int(text1: str, text2: str, rubric: str) -> int:
    prompt = f"""
            Evaluate the two texts below strictly according to the provided rubric.

            Rubric:
            {rubric}

            Text 1:
            {text1}

            Text 2:
            {text2}

            Return one and only one signed integer: the score difference (Text 1 - Text 2). Do not include any explanation, or extra characters. Any output other than a single signed integer is invalid.
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

        result = response.choices[0].message.content.strip()
        return int(result)

    except (ValueError, IndexError, AttributeError) as parse_err:
        raise RuntimeError(f"Failed to parse score difference: {parse_err}")

    except Exception as api_err:
        raise RuntimeError(f"OpenAI API call failed: {api_err}")


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


def predict_scores(test_data: pd.DataFrame, training_data: pd.DataFrame, rubric: str):

    predictions = []
    # renaming for clarity: data = test_data, baseline = training_data

    for i, row_i in test_data.iterrows():
        store_pred_scores = []

        for j, row_j in training_data.iterrows():
            if i <= j:  # type: ignore
                # try <= and >= both, this halfs the squared data, do that as double checking, does full set work better or is half-set sufficient already in obtaining good results,
                # Todo for debugging: isolated predictions, give same pair twice, then see if difference is 0
                continue
            try:
                diff = get_pair_diff_as_int(row_i["essay"], row_j["essay"], rubric)
                score_pred = row_j["domain1_score"] + diff
                store_pred_scores.append(score_pred)
            except RuntimeError:
                continue

        avg_score = 0
        # if the list is empty, it means no valid differences were found
        if len(store_pred_scores) != 0:
            avg_score = sum(store_pred_scores) / len(store_pred_scores)

        predictions.append(avg_score)

    test_data["y_pred"] = predictions
    return test_data


limit = 4  # Limit the number of rows for testing
limit_data = limit  # Limit the number of rows for testing
limit_baseline = 2 * limit  # Limit the number of rows for testing

assert limit > 0, "Limit must be greater than 0."
assert limit < len(X_train), "Limit exceeds number of rows in X_train."
assert limit <= 6, "Limit exceeds number of reasonable rows."

# now include the original target variable from y_train in the prediction of the difference
data_baseline = prelim_reduced_data.tail(limit_baseline)
data_for_pred = prelim_reduced_data.head(limit_data)
rubric = rubric_set_1_text


score_prediction = predict_scores(data_for_pred, data_baseline, rubric)
print(score_prediction)

y_true = data_for_pred["domain1_score"]
y_pred = score_prediction["y_pred"]

mse = mean_squared_error(y_true, y_pred)
print(f"Mean Squared Error: {mse:.2f}")


# todo: cash get_pair_diff_as_int , reduces the API calls, use the FULL string, not only the inputs, because for example the prompt could be modified from one call to the other, cashing as dataframe, string and int diff for querying, printing to see which is a fresh API call and which come form the cash

# todo: include support for the other essay sets (other rubrics, different column names, etc.)

# todo: connect my work to the pre-folded / split data instead of the test-version

# todo: implement a baseline predictor that only predicts the score directly from the model, for comparison

# todo: make sure the model doesn't have access to the internet so it doesn't just look-up

# ! thesis will get registered now, this means an official DEADLINE, I will get an e-mail about that
# new title: Pairwise Difference Learning for LLMs
