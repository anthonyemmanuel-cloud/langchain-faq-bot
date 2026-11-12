"""
GLAB 927.6.1: Mastering LangChain for Advanced Language Model Applications
Student: Anthony Emmanuel

LangChain is a framework that helps you build apps with large language
models (LLMs). It lets you:
    - write reusable PROMPT TEMPLATES (prompts with fill-in-the-blanks)
    - connect steps into CHAINS (the output of one step feeds the next)

This program builds a customer support assistant using LangChain and Groq:
    Task 2: a basic FAQ prompt chain (business hours, return policy, ...)
    Task 3: an advanced 3-step chain (extract -> retrieve -> respond)
    Task 4: the SAME 3-step chain with OPTIMIZED prompts, evaluated on
            several different customer queries

How to run:
    uv run langchain-faq-bot

Setup:
    Put your Groq API key in a file named .env (never in the code!):
        GROQ_API_KEY=your-key-here

Note: the lab PDF uses the model "llama3-8b-8192" and LLMChain. Groq shut
that model down in August 2025, and LangChain 1.0 removed LLMChain, so
this code uses the official replacement model and the modern "pipe" style.
"""

import os
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_groq import ChatGroq

# Groq's official replacement for the retired Llama 8B models
MODEL_NAME = "openai/gpt-oss-20b"

# The project folder (two levels up from this file: src/langchain_faq_bot/)
PROJECT_FOLDER = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------
# Sample order database for Task 4.
# In a real company this would be a database or an API. An LLM cannot
# look up real orders by itself, so the optimized Prompt 2 is told
# exactly WHERE to get the information from: these records.
# ---------------------------------------------------------------
ORDERS_DATABASE = {
    "12345": {
        "status": "Shipped",
        "item": "Wireless Headphones",
        "order_date": "November 3, 2026",
        "carrier": "UPS",
        "tracking_number": "1Z999AA10123456784",
        "estimated_delivery": "November 13, 2026",
    },
    "67890": {
        "status": "Delivered",
        "item": "Ergonomic Office Chair",
        "order_date": "October 28, 2026",
        "carrier": "FedEx",
        "tracking_number": "7489 2233 1100",
        "estimated_delivery": "Delivered on November 2, 2026",
    },
    "24680": {
        "status": "Processing",
        "item": "4K Monitor",
        "order_date": "November 9, 2026",
        "carrier": "Not assigned yet",
        "tracking_number": "Not assigned yet",
        "estimated_delivery": "November 18, 2026",
    },
}


def format_order_records():
    # Turn the database into plain text, one order per line, so it can be
    # placed inside a prompt. Example line:
    # Order 12345 | status: Shipped | item: Wireless Headphones | ...
    lines = []
    for order_number, details in ORDERS_DATABASE.items():
        fields = " | ".join(f"{key}: {value}" for key, value in details.items())
        lines.append(f"Order {order_number} | {fields}")
    return "\n".join(lines)


# ---------------------------------------------------------------
# Setup: load the API key safely and create the LLM
# ---------------------------------------------------------------
def create_llm():
    # Read GROQ_API_KEY from the .env file in the project folder.
    # This keeps the secret key OUT of the code (and off GitHub).
    load_dotenv(PROJECT_FOLDER / ".env")

    if not os.getenv("GROQ_API_KEY"):
        raise SystemExit(
            "Missing GROQ_API_KEY. Create a file named .env in the project "
            "folder containing: GROQ_API_KEY=your-key-here"
        )

    # temperature=0 makes answers more consistent (less random),
    # which is better for customer support.
    # max_retries=5: if Groq's free tier says "too many requests" (error 429),
    # wait and try again up to 5 times instead of crashing.
    # reasoning_effort="low": this model "thinks" before it answers; low keeps
    # it fast and uses fewer tokens (simple support questions don't need more).
    return ChatGroq(model=MODEL_NAME, temperature=0, max_retries=5, reasoning_effort="low")


# ---------------------------------------------------------------
# Task 2: Setting Up LangChain and LangChain-Groq (basic FAQ prompt chain)
# ---------------------------------------------------------------
SYSTEM_MESSAGE = "You are a customer support representative. Your goal is to efficiently handle queries."

FAQ_QUESTIONS = [
    "What are your business hours?",
    "What is your return policy?",
]


def build_faq_chain(llm):
    # Create a prompt template for FAQs.
    # "system" = instructions that set the AI's role and behavior
    # "human"  = the customer's question ({question} is a blank to fill in)
    faq_prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_MESSAGE),
        ("human", "{question}"),
    ])
    # The chain: fill in the prompt -> send to the LLM -> get plain text back
    return faq_prompt | llm | StrOutputParser()


def task_2_basic_faq(llm):
    print("=== Task 2: Basic FAQ Prompt Chain ===")
    faq_chain = build_faq_chain(llm)

    # Reuse the same chain for every frequently asked question
    for question in FAQ_QUESTIONS:
        answer = faq_chain.invoke({"question": question})
        print("Question:", question)
        print("Answer:", answer)
        print()


# ---------------------------------------------------------------
# Task 3: Creating Advanced Prompt Chains (the ORIGINAL lab prompts)
# ---------------------------------------------------------------
def build_original_chains(llm):
    # Step 1: Create the first prompt template to retrieve customer query details
    first_prompt_template = PromptTemplate(
        input_variables=["customer_query"],
        template="Extract the key information from the following customer query: {customer_query}",
    )

    # Step 2: Create the second prompt template to fetch order details based on extracted information
    second_prompt_template = PromptTemplate(
        input_variables=["extracted_info"],
        template="Using the extracted information: {extracted_info}, retrieve the relevant order details.",
    )

    # Step 3: Create the third prompt template to generate a detailed response for the customer
    third_prompt_template = PromptTemplate(
        input_variables=["order_details"],
        template=(
            "Based on the following order details: {order_details}, "
            "generate a detailed response to address the customer's query."
        ),
    )

    # Step 4: Define the chains.
    # The "|" pipe connects the steps: prompt -> LLM -> plain text.
    # (This replaces LLMChain, which was removed in LangChain 1.0.)
    first_chain = first_prompt_template | llm | StrOutputParser()
    second_chain = second_prompt_template | llm | StrOutputParser()
    third_chain = third_prompt_template | llm | StrOutputParser()

    return first_chain, second_chain, third_chain


def task_3_advanced_chain(llm):
    print("=== Task 3: Advanced Prompt Chain (original prompts) ===")
    first_chain, second_chain, third_chain = build_original_chains(llm)

    # Input to start the prompt chain (example customer query)
    initial_input = "Can you help me with the status of my order #12345 placed last week?"

    # Run the first chain to extract key information from the query
    extracted_info = first_chain.invoke({"customer_query": initial_input})
    print("Extracted Info:", extracted_info)
    print()

    # Run the second chain to retrieve order details based on the extracted information
    order_details = second_chain.invoke({"extracted_info": extracted_info})
    print("Order Details:", order_details)
    print()

    # Run the third chain to generate a detailed response based on the order details
    customer_response = third_chain.invoke({"order_details": order_details})
    print("Customer Response:", customer_response)
    print()


# ---------------------------------------------------------------
# Task 4: Optimizing and Evaluating Workflows
# The code structure stays the same as Task 3 (three prompt templates,
# three chains). Only the PROMPTS are improved.
# ---------------------------------------------------------------

# OPTIMIZED Prompt 1: a role, exactly what to extract, and an exact output format
OPTIMIZED_PROMPT_1 = PromptTemplate(
    input_variables=["customer_query"],
    template=(
        "You are a customer support assistant for an online electronics store.\n"
        "Extract the key information from the customer query below.\n"
        "Reply in EXACTLY this format and nothing else:\n"
        "Order number: <digits only, without '#', or NONE if there is no order number>\n"
        "Request: <a short summary of what the customer wants>\n\n"
        "Customer query: {customer_query}"
    ),
)

# OPTIMIZED Prompt 2: specify WHERE to get the information from (the order
# records), and exactly what to return when there is no match.
OPTIMIZED_PROMPT_2 = PromptTemplate(
    input_variables=["extracted_info", "order_records"],
    template=(
        "You are an order lookup assistant. The ORDER RECORDS below are your ONLY "
        "source of information. Never invent or guess order details.\n\n"
        "ORDER RECORDS:\n{order_records}\n\n"
        "Using the extracted information: {extracted_info}\n"
        "Find the order whose number matches EXACTLY.\n"
        "- If the order number is NONE, reply exactly: NO ORDER NUMBER PROVIDED\n"
        "- If no record matches, reply exactly: NOT FOUND\n"
        "- Otherwise, copy the matching record line exactly as written, with no other text."
    ),
)

# OPTIMIZED Prompt 3: add the customer's original question as context,
# and clear rules for tone, length, and missing information.
OPTIMIZED_PROMPT_3 = PromptTemplate(
    input_variables=["order_details", "customer_query"],
    template=(
        "You are a friendly, professional customer support agent for an online electronics store.\n"
        "Customer query: {customer_query}\n"
        "Order details: {order_details}\n\n"
        "Write a reply that addresses the customer's query using ONLY the order details above.\n"
        "Rules:\n"
        "- Do not invent any information that is not in the order details.\n"
        "- If the order details say NOT FOUND, apologize and ask the customer to double-check "
        "the order number or reply with the email address used for the order.\n"
        "- If the order details say NO ORDER NUMBER PROVIDED, politely ask for the order number.\n"
        "- Keep the reply under 120 words.\n"
        "- Sign off as 'Customer Support Team'."
    ),
)


def build_optimized_chains(llm):
    # Same structure as Task 3: three prompts, three chains.
    first_chain = OPTIMIZED_PROMPT_1 | llm | StrOutputParser()
    # .partial() fills in the {order_records} blank ahead of time,
    # so this chain is used exactly like before: only {extracted_info} is needed.
    second_chain = OPTIMIZED_PROMPT_2.partial(order_records=format_order_records()) | llm | StrOutputParser()
    third_chain = OPTIMIZED_PROMPT_3 | llm | StrOutputParser()
    return first_chain, second_chain, third_chain


def run_optimized_chains(first_chain, second_chain, third_chain, customer_query):
    # Run the three chains in order, just like Task 3
    extracted_info = first_chain.invoke({"customer_query": customer_query})
    order_details = second_chain.invoke({"extracted_info": extracted_info})
    customer_response = third_chain.invoke(
        {"order_details": order_details, "customer_query": customer_query}
    )
    return extracted_info, order_details, customer_response


def lookup_is_correct(order_details, expected):
    # Did Step 2 return the right result? Compare without caring about upper/lower case.
    return expected.lower() in order_details.lower()


# Different customer queries, each with the result we EXPECT from Step 2.
# This is a small "labeled dataset": we know the right answer ahead of time,
# so we can measure how accurate the workflow is.
TEST_QUERIES = [
    ("Can you help me with the status of my order #12345 placed last week?", "Shipped"),
    ("Hi, did my order 67890 arrive yet? I can't find the package.", "Delivered"),
    ("When will order #24680 ship? I need the monitor soon.", "Processing"),
    ("Where is my order #99999?", "NOT FOUND"),
    ("I want to know about my recent purchase but I lost my order number.", "NO ORDER NUMBER PROVIDED"),
]


def task_4_optimize_and_evaluate(llm):
    print("=== Task 4: Optimized Prompts and Evaluation ===")
    first_chain, second_chain, third_chain = build_optimized_chains(llm)

    correct = 0
    total_seconds = 0.0
    for number, (query, expected) in enumerate(TEST_QUERIES, start=1):
        start = time.perf_counter()  # start the stopwatch
        extracted_info, order_details, customer_response = run_optimized_chains(
            first_chain, second_chain, third_chain, query
        )
        seconds = time.perf_counter() - start  # stop the stopwatch
        total_seconds += seconds

        is_correct = lookup_is_correct(order_details, expected)
        correct += is_correct

        print(f"--- Test {number} ---")
        print("Customer Query:", query)
        print("Extracted Info:", extracted_info)
        print("Order Details:", order_details)
        print(f"Expected: {expected} -> {'CORRECT' if is_correct else 'WRONG'}")
        print("Customer Response:", customer_response)
        print(f"Time: {seconds:.2f} seconds")
        print()

    accuracy = correct / len(TEST_QUERIES) * 100
    average_seconds = total_seconds / len(TEST_QUERIES)
    print(f"Lookup accuracy: {correct}/{len(TEST_QUERIES)} ({accuracy:.0f}%)")
    print(f"Average time per query: {average_seconds:.2f} seconds")


# ---------------------------------------------------------------
# main(): the "front door" of the app.
# `uv run langchain-faq-bot` calls this function.
# ---------------------------------------------------------------
def main():
    llm = create_llm()
    task_2_basic_faq(llm)
    task_3_advanced_chain(llm)
    task_4_optimize_and_evaluate(llm)


# Lets the file also run directly with:
#   uv run python src/langchain_faq_bot/faq_bot.py
if __name__ == "__main__":
    main()