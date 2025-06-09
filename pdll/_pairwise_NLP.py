import logging
import logging.config

import pandas as pd

import pdll._pairwise_NLP_caching as caching
from pdll._pairwise_NLP_dataprocessing import query_the_api

logging.config.fileConfig("pdll\\\\log\\_logging.conf")
logger = logging.getLogger("result")


def get_pair_diff_as_int(
    essay1: str,
    essay2: str,
    rubric: str,
    model: str,
) -> int:
    prompt = f"""
    You are an expert text comparison assistant.
    
    Task:
    Strictly evaluate the two essays according to the rubric below.
    
    Rules:
    Return only one signed integer: the score difference (Essay 1 score minus Essay 2 score).
    Do NOT include any explanations, comments, extra characters, whitespace, or anything besides a single signed integer.
    Any output other than a single signed integer will be considered invalid.

    Rubric:
    {rubric}

    Essay 1:
    {essay1}
    
    Essay 2:
    {essay2}
    """

    from_cache = caching.lookup_in_cache(model, prompt, True)

    if from_cache is None:
        try:
            pred_score = int(query_the_api(model, prompt))
            caching.new_cache_entry(model, prompt, pred_score, True)
            logger.debug("got score from new prediction")
            return pred_score

        except (ValueError, IndexError, AttributeError) as parse_err:
            raise RuntimeError(
                logger.exception(f"Failed to parse score difference: {parse_err}")
            )

        except Exception as api_err:
            raise RuntimeError(logger.exception(f"OpenAI API call failed: {api_err}"))

    else:
        logger.debug("got score from cache")
        return int(from_cache)


def predict_scores_pairwise(
    test_data: pd.DataFrame,
    training_data: pd.DataFrame,
    rubric: str,
    model: str,
) -> pd.DataFrame:
    caching.load_cache(True)
    logger.info("score prediction started\n")

    predictions = []

    for i, row_i in test_data.iterrows():
        store_pred_scores = []

        for j, row_j in training_data.iterrows():
            logger.info(f"Datapoint {i} & Datapoint {j}")

            # if i <= j:  # type: ignore
            # try <= and >= both, this halfs the squared data, do that as double checking, does full set work better or is half-set sufficient already in obtaining good results,
            # Todo for debugging: isolated predictions, give same pair twice, then see if difference is 0
            # continue
            try:
                diff = get_pair_diff_as_int(
                    row_i["essay"],
                    row_j["essay"],
                    rubric,
                    model,
                )
                logger.info(f"Diff: {diff}")

                score_of_baseline_essay = int(row_j.iloc[1])
                logger.info(f"Score of baseline essay: {score_of_baseline_essay}")

                score_pred = score_of_baseline_essay + diff
                logger.info(f"Score prediction: {score_pred}")

                store_pred_scores.append(score_pred)

            except RuntimeError:
                raise RuntimeError(logger.exception(f"prediction failed for: {row_i}"))

        avg_score = 0
        # if the list is empty, it means no valid differences were found
        if store_pred_scores != []:
            avg_score = int(round(sum(store_pred_scores) / len(store_pred_scores), 0))
        logger.info(f"Avg score for essay {i}: {avg_score}\n")

        predictions.append(int(avg_score))

    test_data["y_pred"] = predictions
    logger.debug("predicted scores pairwise")
    return test_data


# run large scale tests with 300 essays
# for both pairwise and solo
# test = 300, train = 10 for pairwise reference, record EVERYTHING
# then later on you can simulate for smaller train essay numbers from the recorded data for more
#
