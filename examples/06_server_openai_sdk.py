"""Example 6 — use the official OpenAI SDK against the server.

The whole point of the server is OpenAI compatibility: point any OpenAI client at
it and existing code works unchanged.

Install the SDK first:

    pip install openai

Start the server in another terminal:

    python app.py

Then run this from the project root:

    python examples/06_server_openai_sdk.py
"""

from openai import OpenAI

# Point base_url at the server. api_key is required by the SDK but ignored here.
client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

completion = client.chat.completions.create(
    # model picks WHICH model answers: deepseek-chat (fast) or deepseek-expert
    # (stronger, slower). thinking (DeepThink) and search (web) are independent
    # toggles; they ride in extra_body, since they're outside OpenAI's schema.
    model="deepseek-chat",
    messages=[{"role": "user", "content": "My name is Ada. Remember it and say hello."}],
    extra_body={"thinking": False, "search": False},
)
print("Turn 1 Reply:\n", completion.choices[0].message.content)

# conversation_id is outside OpenAI's schema, so the SDK keeps it in model_extra.
extra = getattr(completion, "model_extra", None) or {}
cid = extra.get("conversation_id")
print("\nconversation_id:", cid)

# Turn 2 — continue that conversation by sending the id back via extra_body:
if cid:
    turn2 = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": "What was my name? Just the name."}],
        extra_body={"conversation_id": cid},
    )
    print("\nTurn 2 Reply:\n", turn2.choices[0].message.content)
