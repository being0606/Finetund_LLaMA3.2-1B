import os

import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)


def main():
    # 데이터셋 로드
    dataset = load_from_disk("../data/processed/dataset")

    # 모델 및 토크나이저 로드
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Tokenizer에 pad_token 설정
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 데이터 전처리
    def preprocess_function(examples):
        inputs, targets = [], []
        for conversation in examples["conversations"]:
            for turn in conversation:
                if turn["from"] == "human":
                    inputs.append(turn["value"])
                elif turn["from"] == "gpt":
                    targets.append(turn["value"])
        model_inputs = tokenizer(
            inputs, max_length=256, truncation=True, padding="max_length"
        )
        labels = tokenizer(
            targets, max_length=256, truncation=True, padding="max_length"
        ).input_ids
        labels = [
            [-100 if token == tokenizer.pad_token_id else token for token in label]
            for label in labels
        ]
        model_inputs["labels"] = labels
        return model_inputs

    tokenized_dataset = dataset.map(
        preprocess_function, batched=True, remove_columns=dataset.column_names
    )
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # TrainingArguments 설정
    training_args = TrainingArguments(
        output_dir="../models/",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=5e-5,
        save_strategy="epoch",
        fp16=True,
        logging_dir="../logs/",
        report_to="tensorboard",
    )

    # Trainer 초기화
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    trainer.train()

    # 모델 저장
    model.save_pretrained("../models/fine_tuned_model")
    tokenizer.save_pretrained("../models/fine_tuned_model")


if __name__ == "__main__":
    main()
