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
    limit = 3
    limit_data = limit
    limit_baseline = limit
    reasonable = 8
else:
    limit = 3
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
        y_true = score_prediction["score"]
        y_pred = score_prediction["y_pred"]

        mse = mean_squared_error(y_true, y_pred)
        list_mse.append(mse)
        print(f"MSE of Set: {mse:.2f}")
    else:
        print("No score prediction available.")


stop = 4

# run through all essay sets
for i in range(1, stop):
    main(i)

print("\n\nCache-Stats:\n")
caching.print_cache_stats(True)
caching.print_cache_stats(False)

# evaluation
print("\n\nEvaluation:\n")
for j in range(len(list_mse)):
    print(f"MSE of Set {j+1}: {round(list_mse[j], 2)}")
    gathered_mse = gathered_mse + list_mse[j]

avg_mse = round(gathered_mse / len(list_mse), 2)
print(f"\nAverage MSE: {avg_mse}\n\n")

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
