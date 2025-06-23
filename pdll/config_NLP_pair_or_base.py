# list of openAI models
llms = [
    "gpt-4o-mini",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    # "o3-mini",
    "o4-mini",
    "gpt-4o",
    "gpt-4.1",
]

# Set control variables
is_test_run = True

is_pairwise = False

llm = llms[5]

start_at_essay_set = 1
stop_at_essay_set = 1 + 8

fold_ID = 1
random_seed = 81

if is_pairwise:
    limit_data = 30
    limit_anchors = 10
    limit_reasonable = 60
else:
    limit_data = 30
    limit_anchors = 0
    limit_reasonable = 60
