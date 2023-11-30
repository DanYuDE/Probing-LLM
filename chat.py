from transformers import BertTokenizer, BertModel
from bertviz import head_view

# Load model
model_version = 'bert-base-uncased'
model = BertModel.from_pretrained(model_version, output_attentions=True)
tokenizer = BertTokenizer.from_pretrained(model_version)

# Encode text
text = "Here is the sentence I want to encode"
encoded_text = tokenizer.encode(text, return_tensors='pt')

# Get attention
output = model(encoded_text)
attention = output[-1]

# Visualize
tokens = tokenizer.convert_ids_to_tokens(encoded_text[0])
head_view(attention, tokens)
