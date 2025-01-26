from datasets import load_dataset

def preprocess_data():
    dataset = load_dataset("coastral/korean-writing-style-instruct", split="train[:10%]")
    return dataset

if __name__ == "__main__":
    dataset = preprocess_data()
    dataset.save_to_disk("../data/processed/dataset")