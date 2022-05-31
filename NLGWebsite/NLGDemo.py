import sys
sys.path.append("NLGWebsite")
from transformers import T5Tokenizer, T5ForConditionalGeneration
from server import runNoClass


model = T5ForConditionalGeneration.from_pretrained('/model_5')
tokenizer = T5Tokenizer.from_pretrained('/model_5/tokenizer')

def getOutput(text):
    input_ids = tokenizer(text, add_special_tokens=False, return_tensors="pt", padding=True)["input_ids"]
    print(input_ids)
    output = model.generate(input_ids)
    print(output)
    print(tokenizer.decode(output[0], skip_special_tokens=False, padding=False))
    return tokenizer.decode(output[0], skip_special_tokens=False, padding=False)


runNoClass(getOutput)