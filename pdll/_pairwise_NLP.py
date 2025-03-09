from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd
import openai

SEED = 17


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
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = api_key


def get_pair_diff(text1: str, text2: str) -> str:

    prompt = f"""
    Compare the following two texts and return their differences:
    
    Text 1:
    {text1}
    
    Text 2:
    {text2}
    
    Provide a concise and structured response highlighting the key differences.
    """

    response = openai.chat.completions.create(
        model="gpt-4o-2024-11-20",
        messages=[
            {
                "role": "developer",
                "content": "You are an expert text comparison assistant.",
            },
            {"role": "user", "content": prompt},
        ],
    )

    return response.choices[0].message.content


# Example usage (replace 'your_api_key' with your OpenAI API key):
diff = get_pair_diff("Hello world!", "Hello, ChatGPT!")
print(diff)


# d2 = data.head(9)[data.columns[:3]]
# print(d2)
# print(data.info())
