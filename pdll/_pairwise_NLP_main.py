import codecs
import logging
import logging.config

import pandas as pd
from sklearn.metrics import cohen_kappa_score, mean_squared_error

import pdll._pairwise_NLP as pairwise
import pdll._pairwise_NLP_baseline as baseline
import pdll._pairwise_NLP_caching as caching
import pdll._pairwise_NLP_dataprocessing as data_processing
import pdll._pairwise_NLP_rubricextraction as rubric_extraction
import pdll.config_NLP_pair_or_base as _

logging.config.fileConfig("pdll/log/_logging.conf", encoding="utf-8")
logger = logging.getLogger("result")

# Set control variables
TESTING = _.is_test_run
PAIRWISE = _.is_pairwise
LLMS = _.llms

START = _.start_at_essay_set
STOP = _.stop_at_essay_set

SEED = _.random_seed
FOLD_ID = _.fold_ID

LIMIT_DATA = _.limit_data
LIMIT_ANCHORS = _.limit_anchors
LIMIT_REASONABLE = _.limit_reasonable


logger.debug(f"random seed: {SEED}; fold ID: {FOLD_ID}")

# Load scoring rubrics
scoring_rubrics = rubric_extraction.get_rubric_texts_from_files()


def run_prediction_for(essay_set_ID: int, model: str):
    logger.critical(f"ESSAY SET {essay_set_ID}:\n")

    data_train, data_dev, data_test = data_processing.get_data(
        FOLD_ID,
        essay_set_ID,
        True,
    )

    # conversion to DataFrame
    data_train, data_dev, data_test = data_processing.convert_to_dataframe(
        [data_train, data_dev, data_test]
    )

    if TESTING:
        limit_data = min(LIMIT_DATA, len(data_train))
        limit_reason = LIMIT_REASONABLE
        #! limits data for development and testing
        assert limit_data > 0, logger.warning(
            "Limit for data reduction must be greater than 0."
        )

        # assert limit_data >= limit_reason, logger.warning(
        #     "Limit for data reduction exceeds number of reasonable rows."
        # )
        # use random sample for testing
        data_train, data_dev = (
            data_train.sample(limit_data, random_state=SEED),
            data_dev.sample(LIMIT_ANCHORS, random_state=SEED),
        )
        logger.debug("amount of datapoints was limited due to active testing")

    score_prediction = None
    if PAIRWISE:
        score_prediction = pairwise.predict_scores_pairwise(
            data_train,
            data_dev,
            scoring_rubrics[essay_set_ID],
            model,
        )
    else:
        score_prediction = baseline.predict_scores_solo(
            data_train,
            scoring_rubrics[essay_set_ID],
            model,
        )

    if score_prediction is not None:
        # extracting values for error metrics computation
        y_true = score_prediction["score"]
        y_pred = score_prediction["y_pred"]
        score_prediction["diff"] = y_pred - y_true

        # list full dataset
        pd.options.display.max_rows = None
        pd.options.display.min_rows = 1
        logger.critical(f"\n{score_prediction}\n\n")

        # compute QWK
        qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
        logger.info(f"QWK of Set: {qwk:.5f}")

        # normalizing scores with their max value for better comparison
        data_processing.normalize_score(score_prediction, essay_set_ID)
        # extracting values for error metrics computation
        y_true = score_prediction["score"]
        y_pred = score_prediction["y_pred"]

        # compute MSE
        mse = mean_squared_error(y_true, y_pred)
        logger.info(f"MSE of Set: {mse:.5f}\n")

    else:
        qwk = None
        mse = None
        logger.info("No score prediction available.")

    return mse, qwk


def run_through_all_essay_sets_with(model: str):
    # set up variable for collecting results
    list_mse = []
    list_qwk = []
    gathered_mse = 0
    gathered_qwk = 0

    # set up header with info
    logger.critical(f"LLM: {model}")
    # Set limit of rows for testing
    if PAIRWISE:
        logger.critical("Mode: Pairwise")
        logger.critical(f"Datapoints: {LIMIT_DATA}; Anchors: {LIMIT_ANCHORS}\n")
    else:
        logger.critical("Mode: Solo")
        logger.critical(f"Datapoints: {LIMIT_DATA}\n")
    logger.info(f"Run through essay sets {START} to {STOP - 1}\n")

    # run through all essay sets
    for essay_set in range(START, STOP):
        mse, qwk = run_prediction_for(essay_set, model)
        list_mse.append(mse)
        list_qwk.append(qwk)

    # print caching information
    caching.print_cache_stats(PAIRWISE)

    # repetition of header for info
    logger.critical(f"LLM: {model}")
    # Set limit of rows for testing
    if PAIRWISE:
        logger.critical("Mode: Pairwise")
        logger.critical(f"Datapoints: {LIMIT_DATA}; Anchors: {LIMIT_ANCHORS}\n")
    else:
        logger.critical("Mode: Solo")
        logger.critical(f"Datapoints: {LIMIT_DATA}\n")

    # evaluation of run
    logger.critical("Evaluation:")
    # all MSE in one place
    for j in range(START, STOP):
        logger.critical(f"MSE of Set {j}: {list_mse[j - START]:.5f}")
        gathered_mse += list_mse[j - START]
    avg_mse = gathered_mse / len(list_mse)
    logger.critical(f"Average MSE: {avg_mse:.5f}\n")

    # all QWK in one place
    for j in range(START, STOP):
        logger.critical(f"QWK of Set {j}: {list_qwk[j - START]:.5f}")
        gathered_qwk += list_qwk[j - START]
    avg_qwk = gathered_qwk / len(list_qwk)
    logger.critical(f"Average QWK: {avg_qwk:.5f}\n")

    # marks end of run
    logger.critical(
        "-----------------------------------------------------------------\n\n\n"
    )


for llm in LLMS:
    run_through_all_essay_sets_with(llm)


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
