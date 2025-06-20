# list of openAI models
llms = [
    "gpt-4o-mini",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "o3-mini",
    "o4-mini",
    "gpt-4o",
    "gpt-4.1",
    "o3",
]

# Set control variables
is_test_run = True

is_pairwise = True

llm = llms[0]

start_at_essay_set = 1
stop_at_essay_set = 2

fold_ID = 1
random_seed = 81

if is_pairwise:
    limit_data = 10
    limit_anchors = 5
    limit_reasonable = 15
else:
    limit_data = 50
    limit_anchors = 0
    limit_reasonable = 80
