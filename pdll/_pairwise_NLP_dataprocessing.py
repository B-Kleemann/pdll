import codecs
from pathlib import Path

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
            score = int(tokens[score_index])

            if essay_set == prompt_id or prompt_id <= 0:
                data_x.append(content)
                data_y.append(score)
                prompt_ids.append(essay_set)

    return data_x, data_y, prompt_ids


def get_data(fold_id, prompt_id, as_list_of_tuples):

    dir = Path(f"data-set\\asap\\fold_{fold_id}")
    paths = [f"{dir}\\train.tsv", f"{dir}\\dev.tsv", f"{dir}\\test.tsv"]

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

    if as_list_of_tuples:
        train = list(zip(train_x, train_y))
        dev = list(zip(dev_x, dev_y))
        test = list(zip(test_x, test_y))
    else:
        # as tuple of lists
        train = (train_x, train_y)
        dev = (dev_x, dev_y)
        test = (test_x, test_y)

    return train, dev, test
