import os
from openai import OpenAI

# Hugging Face access token here
HF_API_TOKEN = "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" 

client = OpenAI(
    base_url="https://router.huggingface.co/v1", 
    api_key=HF_API_TOKEN
)

MODEL_ID = "deepseek-ai/DeepSeek-R1"


def main():
    print("=== Anurag Chatbot ===")
    # print(f"Model: {MODEL_ID}")
    print("------------------------------------")
    print("Type your message. Commands: /help /exit")
    print("------------------------------------")

    messages = [
        {"role": "system", "content": "You are a friendly helpful assistant."}
    ]

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue

        if user_input.lower() == "/exit":
            print("Assistant: Bye!")
            break

        if user_input.lower() == "/help":
            print("Assistant: I am a chatbot powered by Anurag Gupta.")
            continue

        messages.append({"role": "user", "content": user_input})

        try:
            response = client.chat.completions.create(
                model=MODEL_ID,
                messages=messages,
                temperature=0.7,
                max_tokens=256,
            )

            assistant_reply = response.choices[0].message.content
            messages.append({"role": "assistant", "content": assistant_reply})

            print("Assistant:", assistant_reply)
            print("------------------------------------")

        except Exception as e:
            print("Error calling API:", e)
            break


if __name__ == "__main__":
    main()
