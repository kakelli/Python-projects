from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from datasets import load_dataset
from transformers import Trainer, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments
from transformers import AutoTokenizer, AutoModelForCausalLM

#Basic Usage(Pipeline API)
'''classifier = pipeline("sentiment-analysis")
print(classifier("I love Caesar"))'''

#Text Classification(Tokenizers and Models)
'''tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased')
inputs = tokenizer("I love using transformers", return_tensors ='pt')
outputs = model(**inputs)
logits = outputs.logits
probs = torch.nn.functional.softmax(logits, dim = -1`)
print(probs)'''

#Training your own model (Fine Tuning)
'''dataset = load_dataset('imdb') #Loading the dataset imdb
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased') #Converting the reviews to arrays
def tokenize_fn(example):
    return tokenizer(example['text'], truncation = True, padding = 'max_length')
encoded_dataset = dataset.map(tokenize_fn, batched = True)
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels = 2) #Loading the model
training_args = TrainingArguments(
    output_dir = "./results",
    evaluation_strategy="epochs",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs = 1,
)
trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = encoded_dataset['train'].select(range(2000)),
    eval_dataset = encoded_dataset['test'].select(range(500))
)
trainer.train()'''

#Text Generation(with gpt)
tokenizer = AutoTokenizer.from_pretrained('gpt-2')
model = AutoModelForCausalLM.from_pretrained('gpt-2')
'''input_ids = tokenizer("Once upon the time", return_tensors= 'pt').input_ids
output = model.generate(input_ids,max_length = 50)
print(tokenizer.decode([0],skip_special_tokens=True))'''

#Custom training loop
'''optimizer = torch.optim.AdamW(model.parameters(), lr = 5e-5)
model.train()
for batch in train_dataloader:
    output = model(**batch)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()'''

#Save and load model
'''model.save_pretrained("my_model")
tokenizer.save_pretrained("my_model")'''

#Model Hub
'''model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")'''
