import logging
import logging.config

import pandas as pd
from sklearn.metrics import mean_squared_error

import pdll._pairwise_NLP as pairwise
import pdll._pairwise_NLP_baseline as baseline
import pdll._pairwise_NLP_caching as caching
import pdll._pairwise_NLP_dataprocessing as data_processing
import pdll._pairwise_NLP_rubricextraction as rubric_extraction

logging.config.fileConfig("pdll\\logging.conf")
logger = logging.getLogger("result")

# Set control variables
TESTING = True
PAIRWISE = False
START_AT_ESSAY_SET = 1
STOP_AT_ESSAY_SET = 8
SEED = 81
FOLD_ID = 1
logger.debug(
    f"pairwise: {PAIRWISE}; testing: {TESTING}; random seed: {SEED}; fold ID: {FOLD_ID}"
)

# Set variables for data
as_list_of_tuples = True

# Load scoring rubrics
scoring_rubrics = rubric_extraction.get_rubric_texts_from_files()

logger.info(f"Run through essay sets {START_AT_ESSAY_SET} to {STOP_AT_ESSAY_SET}")

# Set limit of rows for testing
if PAIRWISE:
    limit = 2
    limit_data = limit
    limit_baseline = limit
    reasonable = 8
    logger.info("Mode: Pairwise\n")
else:
    limit = 4
    limit_data = limit
    limit_baseline = limit
    reasonable = 30
    logger.info("Mode: Solo\n")

# set up variable for collecting results
list_mse = []
gathered_mse = 0


def main(essay_set_ID):
    logger.info(f"ESSAY SET {essay_set_ID}:\n")
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
        assert limit > 0, logger.warning(
            "Limit for data reduction must be greater than 0."
        )
        assert limit <= reasonable, logger.warning(
            "Limit for data reduction exceeds number of reasonable rows."
        )
        # use random sample for testing
        data_train, data_dev, data_test = (
            data_train.sample(limit_baseline, random_state=SEED),
            data_dev.sample(limit_data, random_state=SEED),
            data_test.sample(limit, random_state=SEED),
        )
        logger.info("amount of datapoints was limited due to active testing")

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
        logger.info(f"MSE of Set: {mse:.5f}\n")
        logger.debug(f"\n{score_prediction}\n\n\n")
    else:
        logger.info("No score prediction available.")


for i in range(START_AT_ESSAY_SET, STOP_AT_ESSAY_SET + 1):
    main(i)

caching.print_cache_stats(PAIRWISE)

# evaluation
logger.info("Evaluation:")
for j in range(START_AT_ESSAY_SET, STOP_AT_ESSAY_SET + 1):
    logger.info(f"MSE of Set {j}: {list_mse[j - 1]:.5f}")
    gathered_mse += list_mse[j - 1]

avg_mse = gathered_mse / len(list_mse)
logger.info(f"Average MSE: {avg_mse:.5f}\n")
logger.info(
    "-----------------------------------------------------------------\n\n\n\n\n"
)


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
