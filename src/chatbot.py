from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

def chatbot():
    # 모델과 토크나이저 로드
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    print(f"Loading model: {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # pad_token_id 설정 (없을 경우 eos_token_id 사용)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 파이프라인 생성
    chat_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        truncation=True  # Truncation 활성화
    )
    
    print("\nChatbot is ready! Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Exiting chatbot. Goodbye!")
            break
        
        # 모델 예측
        response = chat_pipeline(
            user_input,
            max_length=100,        # 생성 문장 최대 길이
            do_sample=True,        # 샘플링 활성화
            temperature=0.7        # 샘플링 온도 조정
        )
        print(f"Bot: {response[0]['generated_text']}")

if __name__ == "__main__":
    chatbot()