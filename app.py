# Sample code to get a long sample text and train an LLM QA model on it using the transformers library
import transformers as trf
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Load the model
model_name = "deepset/roberta-base-squad2"
model = trf.AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = trf.AutoTokenizer.from_pretrained(model_name)

# sample text that has 2000 words about a lady who is a teacher and a mother of two kids
text = """
A lady named Mary is a teacher and a mother of two kids. She is a very hardworking person and she loves her job.
She teaches at a local school and she is very passionate about her work. She always tries to give her best to her students.
Mary is also a very loving mother. She
loves her kids more than anything in the world. She always tries to spend quality time with her kids.
She takes them to the park, plays with them, and reads them bedtime stories. Mary is a very caring and responsible mother.
She always makes sure that her kids are happy and healthy. Mary is a very busy person but she always manages to find time for her family.

Mary is a very kind and generous person. She always helps others in need. She volunteers at a local charity and
she donates money to help the less fortunate. Mary is a very positive person and she always sees the good in people.
She believes that everyone deserves a chance and she always tries to make the world a better place.

Mary is a very intelligent person. She is always reading books and learning new things. She is very curious and she
always asks questions. Mary is a very good listener and she always pays attention to what others have to say.
"""

# tokenize the text
inputs = tokenizer(text, return_tensors="pt")
# run the model
outputs = model(**inputs)
# get the answer
answer_start_index = outputs.start_logits.argmax()
answer_end_index = outputs.end_logits.argmax()
answer = tokenizer.decode(
    inputs["input_ids"][0][answer_start_index : answer_end_index + 1]
)
print(answer)

# write a question to ask the model
question = "What is Mary's job?"
# tokenize the question
inputs = tokenizer(question, text, return_tensors="pt")
# run the model
outputs = model(**inputs)
# get the answer
answer_start_index = outputs.start_logits.argmax()
answer_end_index = outputs.end_logits.argmax()
answer = tokenizer.decode(
    inputs["input_ids"][0][answer_start_index : answer_end_index + 1]
)
print(answer)

# Summazise the text
summarizer = pipeline("summarization")
summary = summarizer(text, max_length=100, min_length=30, do_sample=False)[0]

print(summary["summary_text"])

# another question
question = "Who is Mary's mother?"
# tokenize the question
inputs = tokenizer(question, text, return_tensors="pt")
# run the model

outputs = model(**inputs)
# get the answer
answer_start_index = outputs.start_logits.argmax()
answer_end_index = outputs.end_logits.argmax()
answer = tokenizer.decode(
    inputs["input_ids"][0][answer_start_index : answer_end_index + 1]
)
print(answer)
model.save_pretrained("models/roberta-base-squad2")

# load the model
model = trf.AutoModelForQuestionAnswering.from_pretrained("models/roberta-base-squad2")