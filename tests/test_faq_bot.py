"""
Automated tests for the LangChain FAQ bot.

These tests use a FAKE language model that returns answers we choose
ahead of time. That means they run instantly, for free, without
internet or an API key, and give the same result every time.

Run them with:
    uv run pytest
"""

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from langchain_faq_bot.faq_bot import (
    OPTIMIZED_PROMPT_2,
    build_faq_chain,
    build_optimized_chains,
    build_original_chains,
    format_order_records,
    lookup_is_correct,
    run_optimized_chains,
)


# ---------- the order records (where Prompt 2 gets its information) ----------

def test_order_records_include_every_order():
    records = format_order_records()
    assert "Order 12345 | status: Shipped" in records
    assert "Order 67890 | status: Delivered" in records
    assert "Order 24680 | status: Processing" in records


def test_prompt_2_contains_the_order_records():
    prompt_text = OPTIMIZED_PROMPT_2.format(
        extracted_info="Order number: 12345", order_records=format_order_records()
    )
    assert "Wireless Headphones" in prompt_text   # the data really is in the prompt
    assert "NOT FOUND" in prompt_text             # and the rule for missing orders


# ---------- checking results ----------

def test_lookup_is_correct_ignores_upper_and_lower_case():
    assert lookup_is_correct("Order 12345 | status: shipped", "Shipped")


def test_lookup_is_correct_catches_wrong_answers():
    assert not lookup_is_correct("Order 12345 | status: Shipped", "Delivered")


# ---------- Task 2: FAQ chain ----------

def test_faq_chain_answers_a_question():
    fake_llm = FakeListChatModel(responses=["We are open 9 AM to 5 PM."])
    faq_chain = build_faq_chain(fake_llm)
    assert faq_chain.invoke({"question": "What are your business hours?"}) == "We are open 9 AM to 5 PM."


# ---------- Task 3: original chains ----------

def test_original_chains_pass_text_along():
    fake_llm = FakeListChatModel(responses=["extracted", "details", "final reply"])
    first, second, third = build_original_chains(fake_llm)
    assert first.invoke({"customer_query": "Where is #12345?"}) == "extracted"
    assert second.invoke({"extracted_info": "extracted"}) == "details"
    assert third.invoke({"order_details": "details"}) == "final reply"


# ---------- Task 4: optimized chains ----------

def test_optimized_chains_run_in_order():
    fake_llm = FakeListChatModel(responses=[
        "Order number: 12345\nRequest: order status",                  # step 1
        "Order 12345 | status: Shipped | item: Wireless Headphones",  # step 2
        "Your headphones have shipped!",                               # step 3
    ])
    chains = build_optimized_chains(fake_llm)
    extracted, details, reply = run_optimized_chains(*chains, "Where is my order #12345?")
    assert extracted.startswith("Order number: 12345")
    assert lookup_is_correct(details, "Shipped")
    assert reply == "Your headphones have shipped!"


def test_optimized_chains_handle_missing_order():
    fake_llm = FakeListChatModel(responses=[
        "Order number: 99999\nRequest: order status",
        "NOT FOUND",
        "Sorry, we could not find that order.",
    ])
    chains = build_optimized_chains(fake_llm)
    _, details, _ = run_optimized_chains(*chains, "Where is my order #99999?")
    assert lookup_is_correct(details, "NOT FOUND")