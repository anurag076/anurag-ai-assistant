from flask import Flask, render_template, request, redirect
from openai import OpenAI
import csv
from datetime import datetime
import os
from flask import send_file

import re

def remove_think_block(text: str) -> str:
    """
    Removes <think>...</think> blocks from LLM output.
    """
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# ---------- CONFIG ----------
  # <-- put your HF token here
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_API_TOKEN,
)

MODEL_ID = "deepseek-ai/DeepSeek-R1"

app = Flask(__name__)

CSV_FILE = "chat_log.csv"

# ----- MODE-SPECIFIC SYSTEM PROMPTS -----
MODE_SYSTEM_PROMPTS = {
    "general": (
        "You are Anurag AI Assistant, a friendly and knowledgeable trainer in "
        "computer networking, network automation, Python, and cloud. "
        "Explain concepts clearly with simple language and real-world IT examples."
    ),
    "ccna": (
        "You are Anurag AI Assistant, an expert CCNA trainer. "
        "Explain networking fundamentals (OSI, IP, VLANs, routing, ACLs, etc.) "
        "in simple language, step-by-step. Use examples from real Cisco networks "
        "and relate to CCNA exam topics."
    ),
    "automation": (
        "You are Anurag AI Assistant, an expert in network automation. "
        "Focus on Python for network engineers, APIs (REST, NETCONF, RESTCONF), "
        "tools like Netmiko, NAPALM, Ansible, and CI/CD for network configs. "
        "Explain how to automate real-world network tasks."
    ),
    "python": (
        "You are Anurag AI Assistant, a Python trainer for networking and IT students. "
        "Explain Python concepts (syntax, functions, modules, OOP, error handling, etc.) "
        "with simple examples, especially related to networking and automation."
    ),
}

# Global state (OK for single-user/local demo)
current_mode = "general"
messages = []


def init_messages(mode: str):
    """Initialize conversation with the proper system prompt for the selected mode."""
    system_prompt = MODE_SYSTEM_PROMPTS.get(mode, MODE_SYSTEM_PROMPTS["general"])
    return [
        {
            "role": "system",
            "content": system_prompt,
        }
    ]


def log_qa(mode: str, question: str, answer: str):
    """
    Append a Q&A pair to a CSV file with timestamp and mode.
    File: chat_log.csv in the project folder.
    """
    file_exists = os.path.isfile(CSV_FILE)
    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "mode", "question", "answer"])
        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            mode,
            question,
            answer
        ])


@app.route("/", methods=["GET", "POST"])
def chat():
    global messages, current_mode

    if request.method == "POST":
        # Read selected mode from form
        selected_mode = request.form.get("mode", "general")

        # If user changed mode, reset conversation with new system prompt
        if selected_mode != current_mode:
            current_mode = selected_mode
            messages = init_messages(current_mode)

        # Make sure messages is initialized
        if not messages:
            messages = init_messages(current_mode)

        user_input = request.form.get("user_input", "").strip()

        if user_input:
            # Command to clear chat
            if user_input.lower() == "/clear":
                messages = init_messages(current_mode)
                return redirect("/")

            # Add user message
            messages.append({"role": "user", "content": user_input})

            try:
                # Call HF Router (OpenAI-style)
                response = client.chat.completions.create(
                    model=MODEL_ID,
                    messages=messages,
                    max_tokens=300,
                    temperature=0.7,
                )
                raw_reply = response.choices[0].message.content
                assistant_reply = remove_think_block(raw_reply)


                messages.append({"role": "assistant", "content": assistant_reply})

                # ---- Log Q&A to CSV ----
                log_qa(current_mode, user_input, assistant_reply)

            except Exception as e:
                error_text = f"Error calling Anurag AI backend: {e}"
                messages.append({"role": "assistant", "content": error_text})

        return redirect("/")

    # GET request
    if not messages:
        messages = init_messages(current_mode)

    visible_messages = [
        m for m in messages if m["role"] in ("user", "assistant")
    ]
    return render_template(
        "chat.html",
        messages=visible_messages,
        current_mode=current_mode,
    )

@app.route("/download")
def download_csv():
    """Serve the chat_log.csv file for download."""
    if os.path.isfile(CSV_FILE):
        return send_file(
            CSV_FILE,
            mimetype="text/csv",
            as_attachment=True,
            download_name="chat_log.csv"
        )
    else:
        # If the file doesn't exist, return an empty CSV
        return "CSV log file not found. Ask at least one question first!"


if __name__ == "__main__":
    #app.run(debug=True)
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))

