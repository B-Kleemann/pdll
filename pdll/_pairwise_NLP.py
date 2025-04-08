from typing import Any
import pandas as pd
import openai
import os

from pathlib import Path
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
from pdfquery import PDFQuery

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


# Input data file from directory and preprocessing

input_dir = Path(
    "C:/Users/betti/Documents/Daten-lokal/Studium/Bachelorarbeit/asap-aes/data"
)

input_file = input_dir / "training_set_rel3.CSV"

data = pd.read_csv(input_file, header=0, sep=";", encoding="latin-1")

data["essay"] = data["essay"].astype(pd.StringDtype())
data["essay"] = data["essay"].str.strip()


# slice data into the different essay prompt sets and store in dictionary
essay_prompt_set_ID = data.essay_set.unique()

dict_of_essay_sets = {elem: pd.DataFrame() for elem in essay_prompt_set_ID}

for key in dict_of_essay_sets.keys():
    dict_of_essay_sets[key] = data[:][data.essay_set == key]


# establish a simple reduced dataset of only the first essay set for development
prelim_reduced_data = dict_of_essay_sets[essay_prompt_set_ID[0]]

prelim_reduced_data = prelim_reduced_data.dropna(axis=1, how="all")

# print(prelim_reduced_data.essay_id.unique().nonzero())


# train test split

prelim_reduced_data = prelim_reduced_data.sample(frac=1, random_state=SEED)

lables_true = prelim_reduced_data.iloc[:, 3:]
data_stripped = prelim_reduced_data.iloc[:, :3]

X_train, X_test, Y_train, Y_test = train_test_split(
    data_stripped, lables_true, test_size=0.25, random_state=SEED, shuffle=True
)


# difference paired essays


def pair_input_averaging():
    avg = ""
    # predict difference in score
    return avg


def pair_input_predict_diff(x1, x2):
    diff = ""
    # predict difference in score
    return diff


def get_pair_diff_as_int(text1: str, text2: str, rubric: str) -> int:
    prompt = f"""
        Compare the following two texts and return the difference in their score according to the provided rubric.

        Rubric:
        {rubric}

        Text 1:
        {text1}

        Text 2:
        {text2}

        Only return a single integer. No explanation, no additional text.
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


# Example usage (replace 'your_api_key' with your OpenAI API key):
# = get_pair_diff_as_score("Hello world!", "Hello, ChatGPT!")

# diff = get_pair_diff_as_int(
#     X_train.iloc[0].essay, X_train.iloc[1].essay, rubric_set_1_text
# )


# print(Y_train.iloc[0])  # essay)
# print(Y_train.iloc[1])  # essay)
# print(diff)


# input (x1, x2), output (diff)
# save all differences, then average over all the differences to obtain final precicion score

# first, a fuction that takes the train, test, and dev set
# use the train set
# have a nested for loop
# loop over all instances
# inside first loop, have storage for differences
# in inner loop, get the difference between the pairs via get_pair_diff_as_int function
# save the difference in the storage
# after inner loop is done, calculate the average of the differences in storage
# how to scale the differences to the range of the scores in the dataset
# predict final score for each instance


def predict_score_diff(data: pd.DataFrame, rubric: str):

    predictions = []

    for i, row_i in data.iterrows():
        diffs = []
        for j, row_j in data.iterrows():
            if i == j:
                continue
            try:
                diff = get_pair_diff_as_int(row_i["essay"], row_j["essay"], rubric)
                diffs.append(diff)
            except RuntimeError:
                continue

        avg_score = sum(diffs) / len(diffs)

        predictions.append(avg_score)

    data["predicted_score"] = predictions
    return data


score_diff = predict_score_diff(X_train.head(3), rubric_set_1_text)
print(score_diff)

print(Y_train.head(3))
