# LangChain Customer Support FAQ Bot

**GLAB 927.6.1: Mastering LangChain for Advanced Language Model Applications**
Per Scholas AI Prompt Engineering (2026-CAX-255) · Anthony Emmanuel

## What this app does

This program uses **LangChain** and an AI model hosted by **Groq** (`openai/gpt-oss-20b`) to build a customer support assistant in three stages:

1. **Task 2: Basic FAQ prompt chain.** One reusable chain (system prompt + customer question) answers common questions such as business hours and the return policy.
2. **Task 3: Advanced prompt chain.** Three prompts run in a row: **extract** the key information → **retrieve** order details → **write** a response. It uses the original lab prompts.
3. **Task 4: Optimized and evaluated workflow.** The **same three-chain code** with **improved prompts**: a role, exact output formats, rules for missing data, and the order records that Prompt 2 must use as its **only source of information**. It is tested on 5 different customer queries with known correct answers to measure accuracy and speed.

## What problem it solves

Support teams answer the same questions over and over. A LangChain workflow can answer FAQs and order-status questions automatically, so people can focus on harder problems. But an AI must be told **where its information comes from**, or it will make up answers.

## Technologies used

- Python 3.13+
- uv for project and dependency management
- LangChain (`langchain-core` prompt templates and the `|` pipe syntax)
- `langchain-groq` with the model **`openai/gpt-oss-20b`**
- `python-dotenv` to keep the API key out of the code
- `pytest` for automated tests

## Updates from the lab PDF

| Lab PDF | This project | Why |
|---|---|---|
| Model `llama3-8b-8192` | `openai/gpt-oss-20b` | Groq shut down `llama3-8b-8192` on August 30, 2025, and its replacement `llama-3.1-8b-instant` is no longer available on the free plan. Groq names `openai/gpt-oss-20b` as the replacement. |
| `LLMChain` and `.run()` | `prompt \| llm \| StrOutputParser()` and `.invoke()` | LangChain 1.0 removed `langchain.chains` and `langchain.prompts` |
| API key typed into the code | API key in a `.env` file that Git ignores | Keys in code get leaked when the code goes on GitHub |

## How to install dependencies

```bash
uv sync
```

## How to set up your API key

1. Get a free API key at https://console.groq.com (API Keys → Create API Key).
2. Copy `.env.example` to a new file named `.env`:
   ```bash
   cp .env.example .env
   ```
3. Open `.env` and replace `your-key-here` with your key.

`.env` is listed in `.gitignore`, so it is never uploaded to GitHub.

## How to run

```bash
uv run langchain-faq-bot
```

The program makes 20 requests to Groq. If the free tier's per-minute limit is reached, it waits and retries automatically.

## How to run the tests

```bash
uv run pytest
```

The tests use a **fake language model**, so they run without internet or an API key.

## Evaluation: original vs. optimized prompts

| | Original prompts (Task 3) | Optimized prompts (Task 4) |
|---|---|---|
| Prompt 1: extract | "Extract the key information..." gives a free-form paragraph | A role, exactly what to extract, and an exact output format (`Order number:` / `Request:`) |
| Prompt 2: retrieve | Asks the AI to "retrieve" order details without saying from where, so it **invents** them or says it has no access | Gives the AI the **order records** as its **only source**, with exact replies for "not found" and "no order number" |
| Prompt 3: respond | "Generate a detailed response" | Includes the customer's original question, uses **only** the order details, handles missing orders, stays under 120 words, signs off as the team |
| Settings | Default randomness | `temperature=0` for consistent answers and `reasoning_effort="low"` for speed |
| Evaluation | One example | 5 labeled queries (shipped, delivered, processing, unknown order, no order number) with lookup accuracy and the average time per query |

## My results (real run, November 12, 2026)

| | Original prompts (Task 3) | Optimized prompts (Task 4) |
|---|---|---|
| Used real order data? | ❌ No | ✅ Yes |
| Invented anything? | ❌ Yes: items, prices and payment details | ✅ No |
| Handled a missing order? | Not tested | ✅ Test 4 (NOT FOUND) |
| Handled a lost order number? | Not tested | ✅ Test 5 (NO ORDER NUMBER PROVIDED) |
| Lookup accuracy | Not measured | **5/5 (100%)** |
| Average time per query | Not measured | **1.13 seconds** |

**What the original Task 3 chain did:** Step 1 extracted the order number (12345) correctly. But Step 2 had no order data, so the model **hallucinated** an order: a T-shirt, a hoodie and an Amazon gift card, prices, and a payment of $124.96 on a "Visa ending in 4242." The real order is a pair of **Wireless Headphones**. Step 3 then wrote a confident reply ("everything looks good") based on the fake details, and even offered help with "the hoodie's care instructions." The fake card number and the UPS tracking number it used are well-known **sample numbers** from online examples, which shows how a model fills gaps with realistic-looking data it saw in training.

**Task 2 also hallucinated:** the FAQ chain made up business hours, a return policy, an email address and a phone number, because the model was never given the company's real information.

**What the optimized Task 4 chain did:** all 5 test queries returned the correct record or the correct rule, and every reply used only the real order details, stayed short and signed off as the Customer Support Team.

## What I learned

- **An LLM cannot look anything up by itself.** The original Prompt 2 asked the model to "retrieve the relevant order details" without giving it any data, so it makes up details that *sound* real (**hallucination**) or says it can't help. Telling the prompt exactly **where to get the information**, by putting the order records in the prompt, fixes this. This is the basic idea behind **RAG** (Retrieval-Augmented Generation).
- **Specific prompts work better.** A role, exact rules, an exact output format and instructions for edge cases make answers more accurate and predictable.
- **Measure, don't assume.** A small set of labeled test queries shows how often the workflow returns the right order. Automated tests with a fake model check the program logic for free.
- **AI models and libraries change fast.** The lab's model was retired, and so was its first replacement. I switched to Groq's current replacement, `openai/gpt-oss-20b`, after reading Groq's deprecation page and the `model_not_found` error.
- **Never put secrets in code.** API keys belong in environment variables or a `.env` file that is excluded from Git.
