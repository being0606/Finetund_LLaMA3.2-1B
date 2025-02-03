import logging
import os

import torch
import torch.distributed as dist
from accelerate import Accelerator
from datasets import load_from_disk
from transformers import (
    DataCollatorForSeq2Seq,  # 필요에 따라 DataCollatorForLanguageModeling(mlm=False)로 변경 가능
)
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

# --- 분산 초기화 시 "device_ids" 인자 제거 (PyTorch 버전 호환 문제 해결) ---
_old_init_process_group = dist.init_process_group


def new_init_process_group(*args, **kwargs):
    if "device_ids" in kwargs:
        del kwargs["device_ids"]
    return _old_init_process_group(*args, **kwargs)


dist.init_process_group = new_init_process_group
# ---------------------------------------------------------------------------

# 로깅 설정
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(
        self,
        dataset_path: str,
        model_name: str,
        output_dir: str,
        logging_dir: str,
        train_args: dict = None,
    ):
        self.dataset_path = dataset_path
        self.model_name = model_name
        self.output_dir = output_dir
        self.logging_dir = logging_dir

        # 기본 학습 파라미터 (평가 관련 옵션 제거)
        self.train_args = train_args or {
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "num_train_epochs": 3,
            "learning_rate": 5e-5,
            "save_strategy": "epoch",
            "fp16": True,
            "logging_steps": 100,  # 로깅 주기 설정
            "report_to": "tensorboard",
        }

        self.tokenizer = None
        self.model = None
        self.dataset = None  # 원본 Dataset (dict가 아닐 수도 있음)
        self.tokenized_dataset = None  # 전처리 후 Dataset
        self.data_collator = None
        self.training_args = None
        self.trainer = None

        # Accelerator 인스턴스 생성 (분산 학습 시 사용)
        self.accelerator = Accelerator()

        self._setup()

    def _setup(self):
        accelerator = self.accelerator

        # 메인 프로세스 우선: 데이터셋 로딩 (캐시 무시 옵션 추가)
        with accelerator.main_process_first():
            self.dataset = load_from_disk(self.dataset_path)
            logger.info("Dataset loaded from disk.")
            if isinstance(self.dataset, dict):
                logger.info(f"Dataset keys: {list(self.dataset.keys())}")
            else:
                logger.info(f"Dataset columns: {self.dataset.column_names}")

        # 먼저 모델 및 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        logger.info("Model and tokenizer loaded.")

        # 데이터셋 전처리: 모델과 토크나이저 로드 이후에 map() 적용
        if isinstance(self.dataset, dict) and "train" in self.dataset:
            train_dataset = self.dataset["train"].map(
                self._preprocess_function,
                batched=True,
                remove_columns=self.dataset["train"].column_names,
                load_from_cache_file=False,
            )
        else:
            train_dataset = self.dataset.map(
                self._preprocess_function,
                batched=True,
                remove_columns=self.dataset.column_names,
                load_from_cache_file=False,
            )
        logger.info("Dataset tokenized.")
        logger.info(
            f"Train dataset columns after mapping: {train_dataset.column_names}"
        )

        # 데이터 Collator 생성
        self.data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)

        # TrainingArguments 설정 (평가 관련 옵션 완전히 제거, remove_unused_columns 해제)
        self.training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=self.train_args["per_device_train_batch_size"],
            gradient_accumulation_steps=self.train_args["gradient_accumulation_steps"],
            num_train_epochs=self.train_args["num_train_epochs"],
            learning_rate=self.train_args["learning_rate"],
            save_strategy=self.train_args["save_strategy"],
            fp16=self.train_args["fp16"],
            ddp_find_unused_parameters=False,
            logging_dir=self.logging_dir,
            logging_steps=self.train_args["logging_steps"],
            report_to=self.train_args["report_to"],
            remove_unused_columns=False,  # 모델 forward와 일치하도록 설정
        )
        logger.info("Training arguments set.")

        # Trainer 초기화 (평가 데이터셋, compute_metrics 등 제거)
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            data_collator=self.data_collator,
        )
        logger.info("Trainer initialized.")

    def _preprocess_function(self, examples: dict) -> dict:
        """
        각 예제 내에서 'human'과 'gpt' 발화를 추출하여,
        모델 학습에 필요한 "input_ids", "attention_mask", "labels"만 반환합니다.
        """
        inputs, targets = [], []
        for conversation in examples["conversations"]:
            for turn in conversation:
                if turn["from"] == "human":
                    inputs.append(turn["value"])
                elif turn["from"] == "gpt":
                    targets.append(turn["value"])

        # 토크나이즈: 반환되는 값은 dict이며 "input_ids"와 "attention_mask" 포함
        model_inputs = self.tokenizer(
            inputs, max_length=256, truncation=True, padding="max_length"
        )
        labels = self.tokenizer(
            targets, max_length=256, truncation=True, padding="max_length"
        ).input_ids

        # pad token 위치를 -100으로 변경 (loss 계산 시 무시)
        labels = [
            [-100 if token == self.tokenizer.pad_token_id else token for token in label]
            for label in labels
        ]
        # 모델 forward에 필요한 열만 반환: input_ids, attention_mask, labels
        return {
            "input_ids": model_inputs["input_ids"],
            "attention_mask": model_inputs["attention_mask"],
            "labels": labels,
        }

    def train(self):
        logger.info("Starting training.")
        self.trainer.train()
        logger.info("Training completed.")

    def save_model(self, save_path: str = None):
        save_path = save_path or os.path.join(self.output_dir, "fine_tuned_model")
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        logger.info(f"Model and tokenizer saved to {save_path}")

    def run(self):
        self.train()
        self.save_model()


def main():
    dataset_path = "data/processed/dataset"  # DatasetDict가 저장된 경로
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    output_dir = "models/"
    logging_dir = "logs/"
    trainer = ModelTrainer(dataset_path, model_name, output_dir, logging_dir)
    trainer.run()


if __name__ == "__main__":
    main()
