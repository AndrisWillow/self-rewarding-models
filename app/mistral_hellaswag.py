from deepeval.benchmarks import HellaSwag
from deepeval.benchmarks.tasks import HellaSwagTask

from awq import AutoAWQForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from deepeval.models.base_model import DeepEvalBaseLLM

class Mistral7B_AWQ(DeepEvalBaseLLM):
    def __init__(
        self,
        model,
        tokenizer
    ):
        self.model = model
        self.tokenizer = tokenizer

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        model = self.load_model()
        device = "cuda"  # the device to load the model onto

        # Tokenize the input prompt
        model_inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = model_inputs["input_ids"].to(device)

        # Note the length of the input
        input_length = input_ids.shape[1]

        # Move the model to the specified device
        model.to(device)

        # Generate the output
        generated_ids = model.generate(
            input_ids,
            max_new_tokens=1,  # Specify the number of new tokens to generate
            do_sample=True,  # Enable sampling to generate more diverse outputs
            temperature=1, # randomness in sampling (higher temp, more creative, but more random, lower, more predictable), effects logits
            top_p=1, # Limits the set of posible next tokens. Does so by cumulatively selecting the most probable tokens from a prob. distribution until it reaches the limit
            top_k=1,   # Limits the options to the {top_k} most likely options
        )

        # Decode only the newly generated tokens, ignoring the input part
        # Subtract input_length from the generated_ids' length to get only new tokens
        new_tokens = generated_ids[0, input_length:].tolist()  # Get only the new token ids
        new_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        return new_text

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

model_name_or_path = "TheBloke/Mistral-7B-Instruct-v0.2-AWQ" #TheBloke/Mistral-7B-v0.1-AWQ #TheBloke/Mistral-7B-Instruct-v0.2-AWQ #solidrust/Nous-Hermes-2-Mistral-7B-DPO-AWQ

model = AutoAWQForCausalLM.from_quantized(model_name_or_path, fuse_layers=True,
                                          trust_remote_code=False, safetensors=True)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=False)

mistral_7b = Mistral7B_AWQ(model=model, tokenizer=tokenizer)

benchmark = HellaSwag(
    tasks=[HellaSwagTask.TRIMMING_BRANCHES_OR_HEDGES, HellaSwagTask.BATON_TWIRLING, HellaSwagTask.APPLYING_SUNSCREEN, HellaSwagTask.ASSEMBLING_BICYCLE, HellaSwagTask.ARCHERY],
    n_shots=0
)

benchmark.evaluate(model=mistral_7b)
print(benchmark.overall_score)