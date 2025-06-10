# list of openAI models
llms = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
]

# Set control variables
is_test_run = True
is_pairwise = True

llm = llms[1]

start_at_essay_set = 1
stop_at_essay_set = 1

fold_ID = 1
random_seed = 81

if is_pairwise:
    limit_data = 5
    limit_anchors = 3
    limit_reasonable = 10
else:
    limit_data = 15
    limit_anchors = 0
    limit_reasonable = 20
