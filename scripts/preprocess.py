from datasets import DatasetDict, load_dataset


def preprocess_data():
    # 전체 데이터셋 불러오기 (기존 "train" split 사용)
    dataset = load_dataset("coastral/korean-writing-style-instruct", split="train")

    # 1. 전체 데이터셋에서 test를 분리합니다.
    train_valid = dataset.train_test_split(test_size=0.2, seed=42)
    test_dataset = train_valid["test"]
    remaining_dataset = train_valid["train"]

    # 2. 남은 데이터를 train과 validation으로 80:20 비율로 분할합니다.
    splits = remaining_dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = splits["train"]  # 80% of 83.33% ≈ 66.67%
    valid_dataset = splits["test"]  # 20% of 83.33% ≈ 16.67%

    # 3. DatasetDict 형태로 통합하여 반환합니다.
    dataset_dict = DatasetDict(
        {
            "train": train_dataset,
            "validation": valid_dataset,
            "test": test_dataset,
        }
    )

    return dataset_dict


if __name__ == "__main__":
    dataset_dict = preprocess_data()
    dataset_dict.save_to_disk("../data/processed/dataset")
