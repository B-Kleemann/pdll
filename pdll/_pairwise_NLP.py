from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd

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

print(X_train.head(9))


# d2 = data.head(9)[data.columns[:3]]
# print(d2)
# print(data.info())
