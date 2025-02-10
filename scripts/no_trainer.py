import logging
import os

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from datasets import DatasetDict, load_dataset, load_from_disk
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm  # tqdm progress bar
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq

# 로깅 설정
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def preprocess_data():
    """
    원본 데이터셋을 불러와 train/validation/test로 split하고 디스크에 저장합니다.
    """
    dataset = load_dataset("coastral/korean-writing-style-instruct", split="train")
    train_valid = dataset.train_test_split(test_size=0.2, seed=42)
    test_dataset = train_valid["test"]
    remaining_dataset = train_valid["train"]
    splits = remaining_dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = splits["train"]
    valid_dataset = splits["test"]
    dataset_dict = DatasetDict(
        {"train": train_dataset, "validation": valid_dataset, "test": test_dataset}
    )
    dataset_dict.save_to_disk("data/processed/dataset")
    return dataset_dict


def preprocess_function(conversations_batch, few_shot, tokenizer):
    """
    conversations_batch: 리스트, 각 원소는 하나의 예시에 해당하는 대화(turns) 리스트입니다.
    예)
        [
            [ {"from": "human", "value": "안녕하세요."},
              {"from": "gpt", "value": "안녕하세요, 무엇을 도와드릴까요?"}, ... ],
            [ {"from": "human", "value": "오늘 날씨 어때?"},
              {"from": "gpt", "value": "맑습니다."}, ... ],
            ...
        ]
    few_shot가 0보다 크면, 각 대화에서 앞 few_shot 쌍은 데모로 사용하고 마지막 쌍을 실제 입력/타깃으로 사용합니다.
    """
    all_input_ids = []
    all_attention_masks = []
    all_labels = []

    for conversation in conversations_batch:
        human_texts = [
            turn["value"] for turn in conversation if turn["from"] == "human"
        ]
        gpt_texts = [turn["value"] for turn in conversation if turn["from"] == "gpt"]

        if (
            few_shot > 0
            and len(human_texts) >= few_shot + 1
            and len(gpt_texts) >= few_shot + 1
        ):
            # 데모 예시: 처음 few_shot 쌍
            demonstrations = []
            for i in range(few_shot):
                demonstrations.append(
                    f"User: {human_texts[i]}\nAssistant: {gpt_texts[i]}"
                )
            few_shot_prompt = "\n".join(demonstrations)
            # 실제 질의: 마지막 human turn과 그에 대한 gpt 응답
            input_text = f"{few_shot_prompt}\nUser: {human_texts[-1]}"
            target_text = gpt_texts[-1]
        else:
            input_text = " ".join(human_texts)
            target_text = " ".join(gpt_texts)

        tokenized_input = tokenizer(
            input_text, max_length=256, truncation=True, padding="max_length"
        )
        tokenized_target = tokenizer(
            target_text, max_length=256, truncation=True, padding="max_length"
        ).input_ids

        # 패딩 토큰은 -100으로 대체하여 loss 계산 시 무시합니다.
        tokenized_target = [
            -100 if token == tokenizer.pad_token_id else token
            for token in tokenized_target
        ]

        all_input_ids.append(tokenized_input["input_ids"])
        all_attention_masks.append(tokenized_input["attention_mask"])
        all_labels.append(tokenized_target)

    return {
        "input_ids": all_input_ids,
        "attention_mask": all_attention_masks,
        "labels": all_labels,
    }


def build_dataloader(
    dataset, split, few_shot, tokenizer, model, batch_size=8, is_train=True
):
    """
    주어진 split("train", "validation" 등)에 대해 전처리 후 DataLoader를 구성합니다.
    """
    proc_dataset = dataset[split].map(
        lambda examples: preprocess_function(
            examples["conversations"], few_shot, tokenizer
        ),
        batched=True,
        remove_columns=["conversations"],
        load_from_cache_file=False,
    )
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    dataloader = DataLoader(
        proc_dataset,
        batch_size=batch_size,
        shuffle=is_train,
        collate_fn=data_collator,
    )
    return dataloader


def train_model(accelerator, model, optimizer, train_dataloader, writer, num_epochs=3):
    """
    학습 루프를 수행합니다.
    """
    if accelerator.is_main_process:
        logger.info("Starting training.")
    model.train()
    for epoch in range(num_epochs):
        if accelerator.is_main_process:
            progress_bar = tqdm(
                total=len(train_dataloader), desc=f"Epoch {epoch+1}", leave=False
            )
        for step, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            batch = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step()
            if accelerator.is_main_process:
                progress_bar.set_postfix(loss=loss.item())
                progress_bar.update(1)
                if writer is not None:
                    writer.add_scalar(
                        "Loss/train", loss.item(), epoch * len(train_dataloader) + step
                    )
        if accelerator.is_main_process:
            progress_bar.close()
    if accelerator.is_main_process:
        logger.info("Training completed. Saving model...")
        accelerator.wait_for_everyone()  # 동기화
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(output_dir)
        logger.info(f"Model saved in {output_dir}")
        if writer is not None:
            writer.close()


def evaluate_model(accelerator, model, eval_dataloader, tokenizer, split="validation"):
    """
    지정한 split("validation" 또는 "test")에 대해 평가를 수행합니다.
    평가 지표: 평균 Loss, Perplexity(PPL), 그리고 Exact Match (EM)
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    total_em = 0  # Exact Match 카운트
    if accelerator.is_main_process:
        progress_bar = tqdm(
            total=len(eval_dataloader), desc=f"Evaluating {split}", leave=False
        )
    with torch.no_grad():
        for batch in eval_dataloader:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            batch_size = batch["input_ids"].size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            # 모델 생성 결과로 EM 계산
            generated_ids = model.generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=256,
            )
            for pred_ids, label_ids in zip(generated_ids, batch["labels"]):
                # -100 값을 제거한 후 디코딩
                filtered_label_ids = [
                    token for token in label_ids.cpu().tolist() if token != -100
                ]
                pred_text = tokenizer.decode(pred_ids, skip_special_tokens=True).strip()
                label_text = tokenizer.decode(
                    filtered_label_ids, skip_special_tokens=True
                ).strip()
                if pred_text == label_text:
                    total_em += 1
            if accelerator.is_main_process:
                progress_bar.update(1)
    if accelerator.is_main_process:
        progress_bar.close()
        avg_loss = total_loss / total_samples
        ppl = torch.exp(torch.tensor(avg_loss))
        em_score = (total_em / total_samples) * 100.0
        logger.info(
            f"Evaluation on {split}: Loss = {avg_loss:.4f}, PPL = {ppl:.4f}, EM = {em_score:.2f}%"
        )
        return avg_loss, ppl, em_score


if __name__ == "__main__":
    # 데이터 전처리 (최초 한 번 실행)
    dataset_dict = preprocess_data()

    # 실험 조건 목록
    experiments = [
        {
            "desc": "Base model (no few-shot)",
            "use_peft": False,
            "few_shot": 0,
        },
        {
            "desc": "Base model with 3-shot",
            "use_peft": False,
            "few_shot": 3,
        },
        {
            "desc": "PEFT model (no few-shot)",
            "use_peft": True,
            "few_shot": 0,
        },
        {
            "desc": "PEFT model with 3-shot",
            "use_peft": True,
            "few_shot": 3,
        },
    ]

    # 모델 및 학습 관련 설정
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    dataset_path = "data/processed/dataset"
    num_epochs = 3
    batch_size = 4  # 작은 배치 사이즈로 실험

    # Accelerator 인스턴스 (모든 실험에 대해 하나로 사용)
    accelerator = Accelerator()

    for exp in experiments:
        if accelerator.is_main_process:
            logger.info(f"Starting experiment: {exp['desc']}")

        # 실험별 출력 디렉토리 설정
        output_dir = f"models/{exp['desc'].replace(' ', '_')}"
        logging_dir = f"logs/{exp['desc'].replace(' ', '_')}"

        # 토크나이저 및 모델 로드
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # PEFT 적용 여부
        if exp["use_peft"]:
            peft_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.1,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, peft_config)

        # 데이터셋 불러오기
        dataset = load_from_disk(dataset_path)

        # DataLoader 구성 (train 및 evaluation)
        train_dataloader = build_dataloader(
            dataset,
            split="train",
            few_shot=exp["few_shot"],
            tokenizer=tokenizer,
            model=model,
            batch_size=batch_size,
            is_train=True,
        )
        eval_dataloader = build_dataloader(
            dataset,
            split="validation",
            few_shot=exp["few_shot"],
            tokenizer=tokenizer,
            model=model,
            batch_size=batch_size,
            is_train=False,
        )

        # 옵티마이저 생성
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

        # Accelerator로 model, optimizer, dataloader 준비
        model, optimizer, train_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader
        )
        eval_dataloader = accelerator.prepare(eval_dataloader)

        # TensorBoard writer (메인 프로세스에서만)
        writer = (
            SummaryWriter(log_dir=logging_dir) if accelerator.is_main_process else None
        )

        # 학습 수행 (3 에폭)
        train_model(
            accelerator,
            model,
            optimizer,
            train_dataloader,
            writer,
            num_epochs=num_epochs,
        )

        if accelerator.is_main_process:
            logger.info("Evaluating model...")
        evaluate_model(
            accelerator, model, eval_dataloader, tokenizer, split="validation"
        )
