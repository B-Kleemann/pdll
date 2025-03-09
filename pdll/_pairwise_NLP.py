from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd
import openai

SEED = 17


# Input data file from directory and preprocessing

inputdir = Path(
    "C:/Users/betti/Documents/Daten-lokal/Studium/Bachelorarbeit/asap-aes/data"
)

inputfile = inputdir / "training_set_rel3.CSV"

data = pd.read_csv(inputfile, header=0, sep=";", encoding="latin-1")

data["essay"] = data["essay"].astype(pd.StringDtype())
data["essay"] = data["essay"].str.strip()


# slice data into the different essay prompt sets and store in dictionary
essayPromptSetID = data.essay_set.unique()

dictOfEssaySets = {elem: pd.DataFrame() for elem in essayPromptSetID}

for key in dictOfEssaySets.keys():
    dictOfEssaySets[key] = data[:][data.essay_set == key]


# establish a simple reduced dataset of only the first essay set for development
prelimReducedData = dictOfEssaySets[essayPromptSetID[0]]

prelimReducedData = prelimReducedData.dropna(axis=1, how="all")

# print(prelimReducedData.essay_id.unique().nonzero())


# train test split

prelimReducedData = prelimReducedData.sample(frac=1, random_state=SEED)

lablesTrue = prelimReducedData.iloc[:, 3:]
data_stripped = prelimReducedData.iloc[:, :3]

X_train, X_test, Y_train, Y_test = train_test_split(
    data_stripped, lablesTrue, test_size=0.25, random_state=SEED, shuffle=True
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
