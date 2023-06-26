from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("t5-base")

model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
