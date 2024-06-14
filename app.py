# Sample code to get a long sample text and train an LLM QA model on it using the transformers library
import transformers as trf
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM


class LLMQAModel:
    def __init__(self, model_name):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # self.qa = pipeline(
        #     "question-answering", model=self.model, tokenizer=self.tokenizer
        # )
        self.summarizer = pipeline(
            "summarization", model=self.model, tokenizer=self.tokenizer
        )

    def get_answer(self, text, question):
        # use start logits and end logits to get the answer
        inputs = self.tokenizer(question, text, return_tensors="pt")
        # run the model
        outputs = self.model(**inputs)
        # get the answer
        answer_start_index = outputs.start_logits.argmax()
        answer_end_index = outputs.end_logits.argmax()
        answer = self.tokenizer.decode(
            inputs["input_ids"][0][answer_start_index : answer_end_index + 1]
        )
        return answer

    def summarize_text(self, text):
        return self.summarizer(text, max_length=100, min_length=30, do_sample=False)[0][
            "summary_text"
        ]

    # return self.qa({"question": question, "context": text})["answer"]

    def save_model(self, path):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load_model(self, path):
        self.model = AutoModelForCausalLM.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        # self.qa = pipeline(
        #     "question-answering", model=self.model, tokenizer=self.tokenizer
        # )


# Example usage
model_name = "deepset/roberta-base-squad2"
model = LLMQAModel(model_name)

# Sample text
text = """
A lady named Mary is a teacher and a mother of two kids. She is a very hardworking person and she loves her job.
She teaches at a local school and she is very passionate about her work. She always tries to give her best to her students.
Mary is also a very loving mother. She loves her kids more than anything in the world. She always tries to spend quality time with her kids.
She takes them to the park, plays with them, and reads them bedtime stories. Mary is a very caring and responsible mother.
She always makes sure that her kids are happy and healthy. Mary is a very busy person but she always manages to find time for her family.
"""

# Question
question = "Who is Mary's mother?"
answer = model.get_answer(text, question)
print("======================Mary's mother===================")
print("Question: ", question)
print("Answer: ", answer)
print("===================Mary's job======================")
# another question
question = "What is Mary's job?"
answer = model.get_answer(text, question)
print("Question: ", question)
print("Answer: ", answer)
print("=========================================")
# Summarize the text
summary = model.summarize_text(text)
print("================summary=========================")
print(summary)
