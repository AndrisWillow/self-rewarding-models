# ! - I had to comment out an evaluation line that was checking if a panda dataframe contained a line - generator_2, otherwise the eval didn't work
# TODO: Figre out why that was

Usage:
Be in the same directory as the data
Run:
alpaca_eval --model_outputs 'model_outputs.json' --reference_outputs 'reference_model_outputs.json' --annotators_config 'alpaca_eval_gpt4_turbo_fn' 