import multiprocessing
multiprocessing.set_start_method("spawn")

import os
from threading import Thread
from typing import Iterator
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

MAX_MAX_NEW_TOKENS = 2048
DEFAULT_MAX_NEW_TOKENS = 1024

MAX_INPUT_TOKEN_LENGTH = int(os.getenv("MAX_INPUT_TOKEN_LENGTH", "4096"))

model_id = "deepseek-ai/deepseek-coder-33b-instruct"
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.use_default_system_prompt = False

def generate(
    message: str,
    chat_history: list,
    system_prompt: str,
    max_new_tokens: int = 1024,
    temperature: float = 1.0,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.5,
) -> Iterator[str]:
    conversation = []
    if system_prompt:
        conversation.append({"role": "system", "content": system_prompt})
    for user, assistant in chat_history:
        conversation.extend([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}])
    conversation.append({"role": "user", "content": message})

    input_ids = tokenizer.apply_chat_template(conversation, return_tensors="pt")
    if input_ids.shape[1] > MAX_INPUT_TOKEN_LENGTH:
        input_ids = input_ids[:, -MAX_INPUT_TOKEN_LENGTH:]
        print(f"Trimmed input from conversation as it was longer than {MAX_INPUT_TOKEN_LENGTH} tokens.")

    streamer = TextIteratorStreamer(tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(
        input_ids=input_ids,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        repetition_penalty=repetition_penalty,
        eos_token_id=32021
    )
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    outputs = []
    for text in streamer:
        outputs.append(text)
        yield "".join(outputs).replace("", "")

if __name__ == "__main__":
    while True:
        # Get user input for prompts
        message = input("User: ")
        #system_prompt = input("System prompt (press Enter if none): ")

        # Set an empty chat history initially
        chat_history = []

        # Generate response
        for response in generate(message, chat_history):
            print("Bibby Powers Florida USA: ", response)
