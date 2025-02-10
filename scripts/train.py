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


class ModelTrainer:
    def __init__(
        self,
        dataset_path,
        model_name,
        output_dir,
        logging_dir,
        few_shot=0,
        use_peft=True,
    ):
        """
        few_shot: few-shot 예시 수 (예: 3을 입력하면, 각 대화에서 처음 3쌍을 데모로 활용)
        use_peft: True이면 PEFT(Lora)를 적용, False이면 기본 모델 그대로 사용
        """
        self.dataset_path = dataset_path
        self.model_name = model_name
        self.output_dir = output_dir
        self.logging_dir = logging_dir
        self.few_shot = few_shot
        self.use_peft = use_peft

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.dataset = load_from_disk(self.dataset_path)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # PEFT 설정 (옵션)
        if self.use_peft:
            peft_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.1,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, peft_config)

        # 데이터 전처리: "conversations" 컬럼을 few-shot 여부에 따라 전처리
        self.train_dataset = self.dataset["train"].map(
            lambda examples: self._preprocess_function(examples["conversations"]),
            batched=True,
            remove_columns=["conversations"],
            load_from_cache_file=False,
        )
        self.data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=8,
            shuffle=True,
            collate_fn=self.data_collator,
        )

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)

        # Accelerator 적용
        self.accelerator = Accelerator()
        self.model, self.optimizer, self.train_dataloader = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader
        )

        # TensorBoard 설정 (메인 프로세스에서만 생성)
        if self.accelerator.is_main_process:
            self.writer = SummaryWriter(log_dir=self.logging_dir)
        else:
            self.writer = None

    def _preprocess_function(self, conversations_batch):
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
            # 각 대화에서 human과 gpt 메시지를 분리
            human_texts = [
                turn["value"] for turn in conversation if turn["from"] == "human"
            ]
            gpt_texts = [
                turn["value"] for turn in conversation if turn["from"] == "gpt"
            ]

            # few_shot 설정이 있고, 최소 few_shot+1 쌍이 존재하면 few-shot 포맷 사용
            if (
                self.few_shot > 0
                and len(human_texts) >= self.few_shot + 1
                and len(gpt_texts) >= self.few_shot + 1
            ):
                # 데모 예시: 처음 few_shot 쌍
                demonstrations = []
                for i in range(self.few_shot):
                    demonstrations.append(
                        f"User: {human_texts[i]}\nAssistant: {gpt_texts[i]}"
                    )
                few_shot_prompt = "\n".join(demonstrations)
                # 실제 질의: 마지막 human turn과 그에 대한 gpt 응답
                input_text = f"{few_shot_prompt}\nUser: {human_texts[-1]}"
                target_text = gpt_texts[-1]
            else:
                # few_shot이 0이거나 충분한 쌍이 없으면 전체를 단순 연결
                input_text = " ".join(human_texts)
                target_text = " ".join(gpt_texts)

            # 토크나이즈: 동일한 max_length, truncation, padding 설정 적용
            tokenized_input = self.tokenizer(
                input_text, max_length=256, truncation=True, padding="max_length"
            )
            tokenized_target = self.tokenizer(
                target_text, max_length=256, truncation=True, padding="max_length"
            ).input_ids

            # 모델의 loss 계산 시 패딩 토큰은 무시하도록 -100 처리
            tokenized_target = [
                -100 if token == self.tokenizer.pad_token_id else token
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

    def train(self, num_epochs=3):
        if self.accelerator.is_main_process:
            logger.info("Starting training.")
        self.model.train()
        for epoch in range(num_epochs):
            if self.accelerator.is_main_process:
                progress_bar = tqdm(
                    total=len(self.train_dataloader),
                    desc=f"Epoch {epoch+1}",
                    leave=False,
                )
            for step, batch in enumerate(self.train_dataloader):
                self.optimizer.zero_grad()
                batch = {k: v.to(self.model.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                self.accelerator.backward(loss)
                self.optimizer.step()

                if self.accelerator.is_main_process:
                    progress_bar.set_postfix(loss=loss.item())
                    progress_bar.update(1)
                    # TensorBoard 기록
                    if self.writer is not None:
                        self.writer.add_scalar(
                            "Loss/train",
                            loss.item(),
                            epoch * len(self.train_dataloader) + step,
                        )
            if self.accelerator.is_main_process:
                progress_bar.close()
        if self.accelerator.is_main_process:
            logger.info("Training completed. Saving model...")
            self.accelerator.wait_for_everyone()  # 모든 프로세스 동기화
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.save_pretrained(self.output_dir)
            logger.info(f"Model saved in {self.output_dir}")
            if self.writer is not None:
                self.writer.close()

    def evaluate(self, split="validation"):
        """
        지정한 split("validation" 또는 "test")에 대해 평가를 수행합니다.
        평가 지표로 평균 loss, perplexity(PPL), 그리고 Exact Match (EM)를 계산하여 로그에 남깁니다.
        """
        eval_dataset = self.dataset[split].map(
            lambda examples: self._preprocess_function(examples["conversations"]),
            batched=True,
            remove_columns=["conversations"],
            load_from_cache_file=False,
        )
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=8,
            collate_fn=self.data_collator,
        )
        eval_dataloader = self.accelerator.prepare(eval_dataloader)
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        total_em = 0  # Exact Match 카운트
        if self.accelerator.is_main_process:
            progress_bar = tqdm(
                total=len(eval_dataloader), desc=f"Evaluating {split}", leave=False
            )
        with torch.no_grad():
            for batch in eval_dataloader:
                batch = {k: v.to(self.model.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                batch_size = batch["input_ids"].size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

                # Generation을 통해 모델 예측을 얻음
                generated_ids = self.model.generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    max_length=256,
                )
                for pred_ids, label_ids in zip(generated_ids, batch["labels"]):
                    # label_ids에서 -100 값을 제거
                    filtered_label_ids = [
                        token for token in label_ids.cpu().tolist() if token != -100
                    ]
                    pred_text = self.tokenizer.decode(
                        pred_ids, skip_special_tokens=True
                    ).strip()
                    label_text = self.tokenizer.decode(
                        filtered_label_ids, skip_special_tokens=True
                    ).strip()
                    if pred_text == label_text:
                        total_em += 1

                if self.accelerator.is_main_process:
                    progress_bar.update(1)
        if self.accelerator.is_main_process:
            progress_bar.close()
            avg_loss = total_loss / total_samples
            ppl = torch.exp(torch.tensor(avg_loss))
            em_score = (total_em / total_samples) * 100.0
            logger.info(
                f"Evaluation on {split}: Loss = {avg_loss:.4f}, PPL = {ppl:.4f}, EM = {em_score:.2f}%"
            )
            return avg_loss, ppl, em_score

    def run(self, num_epochs=0, do_eval=True):
        """
        num_epochs > 0이면 학습 후 평가를 수행합니다.
        do_eval이 True이면 학습 없이 평가만 진행할 수도 있습니다.
        """
        if num_epochs > 0:
            self.train(num_epochs=num_epochs)
        if do_eval:
            self.evaluate()


if __name__ == "__main__":
    # 데이터 전처리 (최초 한 번 실행)
    dataset_dict = preprocess_data()

    # 네 가지 실험 조건
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

    for exp in experiments:
        if Accelerator().is_main_process:
            logger.info(f"Starting experiment: {exp['desc']}")
        # 실험별 출력 디렉토리 설정
        output_dir = f"models/{exp['desc'].replace(' ', '_')}"
        logging_dir = f"logs/{exp['desc'].replace(' ', '_')}"
        trainer = ModelTrainer(
            dataset_path="data/processed/dataset",
            model_name="meta-llama/Llama-3.2-1B-Instruct",
            output_dir=output_dir,
            logging_dir=logging_dir,
            few_shot=exp["few_shot"],
            use_peft=exp["use_peft"],
        )
        # 학습 3 에폭 진행 후 평가 (학습 후 평가를 원하면 아래 주석 해제)
        trainer.train(num_epochs=3)
        if Accelerator().is_main_process:
            logger.info("Evaluating model...")
        trainer.evaluate(split="validation")
