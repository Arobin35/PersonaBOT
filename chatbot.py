import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

# Load your conversation dataset from 'dialogues.json'
with open("dialogues.json", "r") as f:
    data = json.load(f)

# Create a Hugging Face Dataset from the data
dataset = Dataset.from_list(data)

# Specify the model name (DialoGPT-medium)
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Set the pad token to the eos token to enable padding
tokenizer.pad_token = tokenizer.eos_token

# Define a tokenization function that works with batched input
def tokenize_function(examples):
    texts = [
        context + " " + response + tokenizer.eos_token
        for context, response in zip(examples["context"], examples["response"])
    ]
    # Tokenize the texts
    encoding = tokenizer(texts, truncation=True, padding="max_length", max_length=512)
    # For language modeling, labels are typically the same as the input_ids
    encoding["labels"] = encoding["input_ids"].copy()
    return encoding

# Apply tokenization to the dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Define training arguments (fp16 removed for compatibility on MPS)
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    evaluation_strategy="no",  # Set to "no" if you don't have a validation set
    save_steps=500,
    save_total_limit=2,
    logging_steps=100,
    learning_rate=5e-5
)

# Create a Trainer instance for fine-tuning
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer
)

# Start the training process
trainer.train()
