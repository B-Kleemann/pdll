import pandas as pd
from sklearn.metrics import mean_squared_error

import pdll._pairwise_NLP as pairwise
import pdll._pairwise_NLP_baseline as baseline
import pdll._pairwise_NLP_caching as caching
import pdll._pairwise_NLP_dataprocessing as data_processing
import pdll._pairwise_NLP_rubricextraction as rubric_extraction

# Set control variables
TESTING = True
PAIRWISE = True
SEED = 81
FOLD_ID = 1

# Set variables for data
as_list_of_tuples = True

# Load scoring rubrics
scoring_rubrics = rubric_extraction.get_rubric_texts_from_files()

# Set limit of rows for testing
if PAIRWISE:
    limit = 4
    limit_data = limit
    limit_baseline = limit
    reasonable = 8
else:
    limit = 4
    limit_data = limit
    limit_baseline = limit
    reasonable = 30

assert limit > 0, "Limit must be greater than 0."
# assert limit < len(data_train), "Limit exceeds number of rows in dataset."
assert limit <= reasonable, "Limit exceeds number of reasonable rows."

# set up variable for collecting results
list_mse = []
gathered_mse = 0


def main(essay_set_ID):
    print(f"ESSAY SET {essay_set_ID}:\n")
    data_train, data_dev, data_test = data_processing.get_data(
        FOLD_ID,
        essay_set_ID,
        as_list_of_tuples,
    )

    # conversion to DataFrame
    data_train, data_dev, data_test = data_processing.convert_to_dataframe(
        [data_train, data_dev, data_test]
    )

    if TESTING:
        #! limits data for development and testing
        # use random sample for testing
        data_train, data_dev, data_test = (
            data_train.sample(limit_baseline, random_state=SEED),
            data_dev.sample(limit_data, random_state=SEED),
            data_test.sample(limit, random_state=SEED),
        )

    score_prediction = None
    if PAIRWISE:
        score_prediction = pairwise.predict_scores_pairwise(
            pd.DataFrame(data_dev),
            pd.DataFrame(data_train),
            scoring_rubrics[essay_set_ID],
        )
    else:
        score_prediction = baseline.predict_scores_solo(
            pd.DataFrame(data_dev),
            scoring_rubrics[essay_set_ID],
        )

    # Print results
    # print(score_prediction)

    if score_prediction is not None:
        # Compute and print error metrics
        data_processing.normalize_score(score_prediction, essay_set_ID)

        y_true = score_prediction["score"]
        y_pred = score_prediction["y_pred"]

        score_prediction["diff"] = y_true - y_pred

        mse = mean_squared_error(y_true, y_pred)
        list_mse.append(mse)
        print(f"MSE of Set: {mse:.5f}\n")
        print(score_prediction)
        print("\n\n")
    else:
        print("No score prediction available.")


stop = 9
# run through all essay sets
for i in range(1, stop):
    main(i)

print("\nCache-Stats:\n")
caching.print_cache_stats(PAIRWISE)

# evaluation
print("\n\nEvaluation:\n")
for j in range(len(list_mse)):
    print(f"MSE of Set: {list_mse[j]:.5f}")
    gathered_mse = gathered_mse + list_mse[j]

avg_mse = gathered_mse / len(list_mse)
print(f"\nAverage MSE: {avg_mse:.5f}\n\n")


# * DONE
# include support for the other essay sets (other rubrics)
# connect my work to the pre-folded / split data instead of the test-version

# make sure the model doesn't have access to the internet so it doesn't just look-up;
# api calls do not have native access to the internet

# with increasing complexity, bug detection might be more difficult
# include separation between modifications that should and shouldn't affect the score (for example cashing = no impact on score, prompt change = impact on score)
# implement a baseline predictor that only predicts the score directly from the model, for comparison

# cash get_pair_diff_as_int , reduces the API calls, use the FULL string, not only the inputs, because for example the prompt could be modified from one call to the other, cashing as dataframe, string and int diff for querying, printing to see which is a fresh API call and which come form the cash

# * TODO

# ! thesis will get registered now, this means an official DEADLINE, I will get an e-mail about that
# new title: Pairwise Difference Learning for LLMs
