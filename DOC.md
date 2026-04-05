Here is the ultimate, upgraded folder structure for Nyaya-Sahayak, meticulously redesigned to prioritize your Databricks Free Edition constraints, Databricks Apps deployment, Agentic AI routing, and the rigorous MLflow benchmarking requirements discussed.

nyaya_sahayak/
├── app.yaml                            # Mandatory Databricks App config (defines 'streamlit run' command, env vars)
├── requirements.txt                    # Explicit dependencies: streamlit, llama-cpp-python, faiss-cpu, mlflow[databricks], ragas
├── README.md                           # Hackathon mandatory: architecture diagram, execution commands, 1-2 sentence summary
├──.env                                # Local dev environment variables (OAuth tokens, Databricks Connect config)
│
├── data/                               # Managed via Databricks Unity Catalog Volumes / DBFS 
│   ├── bronze/                         # Raw PDFs (Constitution), JSONs (myScheme), CSVs (IPC-to-BNS mapping)
│   ├── silver/                         # PySpark scripts for data cleansing, handling missing values, and chunking
│   └── gold/                           # Z-ordered Delta tables (welfare schemes) & serialized FAISS CPU indices 
│
├── models/
│   ├── llm/
│   │   ├── param1_runner.py            # CPU inference wrapper using llama-cpp-python for Param-1 Q8_0 GGUF 
│   │   └── model_loader.py             # Utility to rapidly load quantized.gguf weights from UC Volumes to RAM
│   ├── translation/
│   │   └── indictrans2_runner.py       # CPU-optimized wrapper for ai4bharat/indictrans2-en-indic-1B 
│   └── embeddings/
│       └── embedder.py                 # Lightweight sentence embedding generator for building the FAISS index
│
├── agents/
│   ├── init.py
│   ├── orchestrator.py                 # Core routing agent: classifies intent and delegates to specific tools
│   ├── tools/                          # Agentic toolkit for dynamic execution
│   │   ├── sql_welfare_tool.py         # Translates demographics into deterministic PySpark queries on Delta Tables
│   │   └── faiss_legal_tool.py         # Performs similarity search on the local FAISS DBFS index 
│   ├── legal_agent.py                  # RAG synthesizer for BNS/BNSS/BSA queries based on retrieved context
│   └── guardrail_agent.py              # Ensures the AI does not offer definitive legal counsel outside its scope
│
├── evaluation/                         # Dedicated MLflow & Benchmarking Suite
│   ├── init.py
│   ├── bhashabench_eval.py             # Script to run BhashaBench-Legal datasets through the pipeline
│   ├── ragas_metrics.py                # Defines MLflow RAGAS scorers: Faithfulness, AnswerAccuracy, ContextPrecision
│   ├── trajectory_eval.py              # Evaluates the orchestrator's decision-making (did it pick the right tool?)
│   └── latency_tracker.py              # @mlflow.trace decorators to measure TTFT (Time To First Token) and TPOT
│
├── notebooks/                          # For execution directly on the Databricks Free Edition Workspace
│   ├── 01_data_ingestion.py            # Uploads data to Volumes and sets up the Medallion architecture
│   ├── 02_delta_faiss_prep.py          # Chunks legal text, creates Delta Tables, and saves FAISS index to DBFS 
│   ├── 03_model_download.py            # Fetches GGUF and IndicTrans2 weights directly into Databricks Volumes
│   └── 04_mlflow_evaluation.py         # Runs the LLM-as-a-judge benchmarking suite
│
└── app/                                # Streamlit Frontend deployed natively via Databricks Apps 
├── main.py                         # Streamlit entry point (referenced by app.yaml) 
├── state_manager.py                # Manages multi-turn conversation history and session state
└── components/
├── chat_view.py                # Conversational UI for the Legal Navigator (Nyaya)
├── scheme_wizard.py            # Widget-based demographic input form for the Welfare Agent (Sahayak)
└── performance_dashboard.py    # UI panel to display MLflow traces, TTFT speeds, and RAGAS accuracy scores

Key Architectural Upgrades Made:
Databricks Apps Native Structure: The root directory now explicitly contains app.yaml and requirements.txt. This is a mandatory pattern for deploying your frontend directly onto Databricks serverless compute without relying on external hosting.

Dedicated Evaluation Module: A robust evaluation/ directory has been integrated to utilize MLflow 3.0. It includes ragas_metrics.py to calculate Faithfulness and ContextPrecision to prove your app doesn't hallucinate Indian laws. The latency_tracker.py uses MLflow tracing to log your Time To First Token (TTFT) and Time Per Output Token (TPOT), proving your CPU optimization works.

Agentic Tool Directory (agents/tools/): Instead of hardcoding the logic, the agents now act as true orchestrators that select tools. The sql_welfare_tool.py is called for deterministic queries against structured Delta Tables, while the faiss_legal_tool.py queries the in-memory vector index stored on DBFS.

Volume/DBFS Data Tracking: The data/ structure explicitly notes the use of Databricks Volumes/DBFS for storing both the raw data and the localized .faiss indices, allowing rapid in-memory loading without OOM errors.


Here’s a unified, detailed project structure and final architecture that merges:

Your Nyaya-Sahayak hackathon plan

The Nyaya-Sahayak agentic pipeline

Local, quantized, CPU-only LLMs and IndicTrans2

Multi‑LLM comparison, act‑specific querying, and a two‑phase UI strategy

It is explicitly designed to fit Databricks Free Edition constraints and the hackathon’s judging rubric.

1. System overview (what you’re building)
Name: Nyaya-Sahayak: Nyaya-Sahayak Lakehouse

Core idea:
A Databricks-native, agentic legal + welfare assistant that:

Ingests BNS/BNSS/BSA/Constitution + IPC→BNS mapping + gov_myscheme data into a Medallion (Bronze/Silver/Gold) lakehouse.

Uses quantized local LLMs (Param‑1 2.9B, optionally Airavata) plus IndicTrans2 for multilingual RAG on CPU.

Routes queries through an agentic pipeline: orchestrator → legal agent → welfare agent → translation agent, with an optional evaluation/comparison agent.

Supports act‑specific queries (“only BNS”, “only BNSS”) and multi‑LLM answer comparison for research/evaluation mode.

Starts with a simple Streamlit/Databricks App UI; later can be wrapped in a more advanced JS frontend.


3. Data / Medallion architecture (on Databricks)
3.1 Bronze (raw) – DBFS / Volumes
Logical mapping (you’ll physically store these under DBFS or Volumes, but track them conceptually here):

data/bronze/laws/ – raw PDFs/JSON

BNS 2023, BNSS 2023, BSA, Constitution, key acts.

data/bronze/ipc_bns_mapping/ – CSV/JSON mapping sources.

data/bronze/schemes/ – gov_myscheme dumps + ancillary scheme PDFs.

data/bronze/legal_aid/ – NALSA/Nyaaya texts & state‑wise rules.

data/bronze/eval/ – BhashaBench-Legal datasets.

data/bronze/synthetic_profiles/ – demo user profiles (age/state/income/caste/gender) for wizard testing.

In 01_data_ingestion.ipynb, use PySpark to ingest these into raw Delta tables (e.g., laws_raw, schemes_raw) without heavy transformation.

3.2 Silver (cleaned & structured)
Transformations in 02_data_cleansing_chunking.ipynb:

Laws:

Parse BNS/BNSS/BSA/Constitution into sections/articles with regex + heuristics.

Table law_sections with columns:

id, code (BNS/BNSS/BSA/CONST/etc.), chapter, section_no, title, text, language, source_url, act_type.

IPC→BNS mapping:

Normalize mapping tables to ipc_bns_map:

ipc_section, bns_section, change_type, notes, source.

Schemes:

Clean gov_myscheme data into schemes_curated:

scheme_id, name, category, description, eligibility_text, age_min, age_max, income_max, gender, caste_flags, state_codes, benefits_text, apply_url.

Legal aid rules:

legal_aid_rules:

state_code, income_ceiling, eligible_categories, law_reference.

Eval:

bb_legal_questions:

id, question_text, options, correct_option, language, domain, difficulty.

3.3 Gold (feature-ready for RAG)
Transformations in 03_embedding_faiss_indexing.ipynb:

Chunking:

Chunk law_sections into ~512-token passages with 50-token overlap; table law_chunks:

chunk_id, section_id, code, section_no, text_chunk, lang, act_type.

Embeddings:

Use models/embeddings/embedder.py to compute embeddings (CPU-optimised sentence-transformer) for each chunk.

Save in law_chunks_embedded:

chunk_id, embedding_en, embedding_hi, metadata.

FAISS indexes:

Build FAISS indexes per domain:

legal_index (laws + constitution)

scheme_index (schemes_curated: name + description + eligibility_text)

Persist indexes to DBFS (e.g., /Volumes/.../faiss/legal.index) and register their paths in config.py.

Z‑ordering:

For schemes_curated, Z‑order on state_codes, age_min, income_max to accelerate filters in welfare agent.

4. Model and quantization architecture
4.1 LLMs (CPU‑only, quantized)
In models/llm/:

Param‑1 2.9B (core reasoning model):

Converted to GGUF Q8_0 or similar via 04_model_quantization.ipynb, using python + llama.cpp tooling.

param1_runner.py wraps llama.cpp (via llama-cpp-python) with:

generate(prompt, max_tokens, temperature, ...)

estimate_memory() to respect ~15 GB limit.

Airavata (Hindi instruction‑tuned model; optional):

Similarly quantized; wrapped in airavata_runner.py for Hindi-heavy citizen queries.

Model router (router.py):

Takes canonical_language, persona, complexity, and decides:

Default: Param‑1.

Hindi + citizen: Airavata (if available) otherwise Param‑1.

Future: pluggable for more models.

4.2 Multi‑LLM comparison (comparator.py)
API:

run_dual_models(prompt, models=["param1", "airavata"]) -> {model_id: answer, comparison: analysis}

Optional “referee” step:

Use Param‑1 with a meta‑prompt: “Given Answer A, Answer B, and the legal context, which is more accurate and why?”

This is used for:

Dev/eval mode in UI.

BhashaBench experiments (e.g., Param‑1 vs Airavata performance in Hindi).

4.3 Translation: IndicTrans2
In agents/translation_agent.py:

Use IndicTrans2 with CTranslate2 CPU backend (distilled checkpoint for efficiency).

Provide:

to_canonical(text) -> (canonical_text, canonical_lang) (map anything to en or hi).

from_canonical(text, target_lang).

5. Agent layer architecture
5.1 Orchestrator (agents/orchestrator.py)
Responsibilities:

Accept UserQuery (from Pydantic model in core/data_models.py):

fields: text, user_lang, persona (citizen/junior_lawyer), act_filter (ALL/BNS/BNSS/BSA/CONST), compare_models (bool).

Call Translation Agent to canonicalise if needed.

Call Intent Classifier (models/nlp_classifier/intent_classifier.py) to decide:

LEGAL, WELFARE, MIXED, or GENERIC.

Route to Legal Agent or Welfare Agent (or both for MIXED).

Optionally call Guardrail Agent to check content safety/scope.

If compare_models is true, invoke multi‑LLM comparison via models/llm/comparator.py.

Format final Answer object with:

answer_text, citations (sections/schemes), confidence_score, model_ids_used.

5.2 Legal agent (agents/legal_agent.py)
Responsibilities:

Use rag/retriever.py to query legal_index, with act filters:

If act_filter != "ALL", limit candidate chunks to that code before FAISS search.

For IPC→BNS queries:

Parse IPC section from query, use ipc_bns_map to fetch mapped sections, then retrieve both IPC and BNS chunks.

Build prompts using rag/prompts.py:

Persona‑aware templates (citizen vs lawyer).

Strict instructions about citing section numbers and not hallucinating outside context.

Call model router → LLM(s) → produce structured LegalAnswer with citations and explanation.

5.3 Welfare agent (agents/welfare_agent.py)
Responsibilities:

If query is conversational: embed question and search scheme_index.

If user has gone through wizard (user profile available), filter schemes_curated by demographics, then rank via embeddings.

Generate explanation with LLM:

“Why this scheme”, “Eligibility criteria summary”, “How to apply”.

Use legal_aid_rules to decide if free legal aid applies and attach Tele‑Law/Nyaya Bandhu links where applicable.

5.4 Translation agent (agents/translation_agent.py)
As above: encapsulates IndicTrans2 for both directions; orchestrator always interacts with canonical text and passes back user‑language response.

5.5 Evaluation agent (agents/evaluation_agent.py)
Responsibilities:

Run BhashaBench‑Legal subsets or custom evaluation sets:

For each question: run RAG pipeline, map answer to MCQ option, compute accuracy.

Possibly evaluate both Param‑1 and Airavata and log to MLflow.

Provide summary stats for a “Model performance” view in the UI.

5.6 Guardrail agent (agents/guardrail_agent.py)
Responsibilities:

Classify/regex‑filter queries that:

Ask how to commit crimes or evade law.

Seek personal legal advice rather than generic information.

Insert disclaimers & redirect to human legal aid where necessary.

6. RAG & retrieval design
In rag/retriever.py and rag/pipeline.py:

Act‑specific retrieval:

Accept act_filter, restrict search to chunks where code == act_filter (or subset) before FAISS search.

Two‑level retrieval:

Primary RAG on statutes/Constitution.

Secondary retrieval on schemes or legal aid, triggered if intent = WELFARE or if LegalAnswer references rights triggering entitlements.

Confidence scoring:

Use max similarity score and coverage of retrieved sections to compute a simple “High / Medium / Low confidence” measure shown in UI.

7. UI strategy: simple first, then advanced
7.1 Phase 1 – Simple Databricks/Streamlit UI (hackathon deliverable)
Dev console notebook (07_dev_console_ui.ipynb):

Widgets for query, persona, language, act filter, compare_models flag.

Uses orchestrator directly and prints:

Answer text

Retrieved sections

Optional Model A vs Model B answers side‑by‑side.

Streamlit app (app/main.py):

Tabs:

Chat (Citizen)

Chat (Junior Lawyer)

Schemes Wizard

IPC→BNS Explorer

Model Comparison / Evaluation (for demo)

Uses components chat_view.py, scheme_wizard.py, ipc_bns_explorer.py, model_compare_panel.py to call orchestrator and agents.

Deployed as a Databricks App using deployment/app.yaml and following Nyaya Dhwani & Apps Cookbook patterns.

7.2 Phase 2 – Robust JS frontend (post‑hackathon)
API layer (api/server.py) exposing:

/chat, /schemes, /ipc_bns, /compare_models, /eval.

JS frontend (React/Next.js) consuming these endpoints with richer UX, but same backend code.

8. Implementation & hackathon execution plan (condensed)
Day 1
Get Bronze → Silver working for a small subset (few BNS chapters + some schemes).

Implement basic RAG pipeline with one model (Param‑1) and simple notebook UI.

Set up Streamlit skeleton and wire simplest chat.

Overnight
Scale ingestion to full BNS + chosen acts and a useful slice of gov_myscheme.

Compute embeddings + FAISS indexes.

Integrate IndicTrans2 and finalize orchestrator + legal/welfare agents.

Add act filter + IPC→BNS mapping for key sections.

Day 2
Harden prompts, implement guardrails & confidence scoring.

Implement schemes wizard and complete welfare agent logic.

Run BhashaBench‑Legal subset via evaluation agent and log metrics.

Hook up multi‑LLM comparison for at least a few test queries.

Polish Streamlit UI + record demo; ensure README + architecture diagram + run commands are ready.

This combined architecture keeps you strongly aligned with the Databricks and hackathon requirements, showcases serious systems thinking (agentic routing, medallion lakehouse, quantized local LLMs), and gives you clear hooks for multi‑LLM comparison, act‑scoped queries, and a path to a full product later.

Prepared by Deep Research
Please explain in detail how the app is used, starting from data ingestion, and processing, and how medallion architecture is used, and how the models are used and whwre they run, also if I am using databricks free edition, then how can I develop locally and use models ? also can agentic ai be used here and how? and how can I pusblish UI app? and load datasets, explain the entire flow/working/building steps in detail

Below is the full lifecycle: how data and models flow through the system, how the Medallion architecture is used, how agentic AI fits in, and how you work both locally and on Databricks Free Edition all the way to a deployed UI app.

1. Big picture: from raw data to user answer
Conceptually, every user query goes through this loop:

Data prepared once

Laws + Constitution + IPC→BNS mapping + gov_myscheme + legal-aid rules are ingested into Databricks, cleaned, chunked, embedded, and indexed (FAISS) following Bronze → Silver → Gold Medallion layers.

User queries

User types (or later speaks) a question into the UI (Streamlit/Databricks App).

Agentic pipeline

Orchestrator agent receives the query, classifies intent, calls translation agent if needed, then routes to legal or welfare agents which use RAG over the Medallion data and local LLMs to generate an answer.

Models

A quantized local LLM (Param‑1, maybe Airavata) runs in the same Python process (Databricks cluster or local dev), reading context from FAISS/Delta and generating a response.

Response back to UI

Answer is (optionally) translated back to the user’s language and rendered with citations, scheme recommendations, confidence score, and optionally comparisons between two LLMs.

Everything else is details on how you set up each piece.

2. Data ingestion & Medallion architecture (Databricks side)
2.1 Bronze: raw ingestion
Goal: Land all external data in DBFS/Volumes as is, and register minimal “raw” Delta tables.

Data sources:

BNS 2023, BNSS 2023, BSA, Constitution PDFs/JSON from official portals.

IPC→BNS mapping CSV (e.g., Hugging Face datasets, govt comparison tables).

gov_myscheme or myScheme-derived datasets for schemes.

Legal-aid rules (NALSA/Nyaaya docs) and BhashaBench‑Legal evaluation data.

How you do it:

In 01_data_ingestion.ipynb, use PySpark to read each source:

spark.read.format("binaryFile") or Python libs for PDFs.

spark.read.csv/json for tabular data.

Write to Delta as raw tables (no heavy logic):

laws_raw, ipc_bns_raw, schemes_raw, legal_aid_raw, bb_legal_raw.

Files stay in DBFS (immutable history), tables in the Bronze schema.

At this stage you’ve only centralised data; no transformations yet.

2.2 Silver: cleaning, structuring, chunking
Goal: Produce clean, structured, queryable tables the rest of the system can depend on.

Key steps in 02_data_cleansing_chunking.ipynb:

Laws:

Parse BNS/BNSS/BSA/Constitution texts by chapters/sections/subsections using regex and NLP heuristics, preserving legal boundaries and adding overlapping context windows (e.g., ~512 tokens with 50-token overlap).

Resulting table law_sections with columns:

id, code (BNS/BNSS/BSA/CONST), chapter, section_no, title, text, language, act_type, source_url.

IPC→BNS mapping:

Clean up mapping data into ipc_bns_map with ipc_section, bns_section, change_type, notes, source.

Schemes & legal-aid:

Clean gov_myscheme into schemes_curated with explicit age_min/max, income_max, gender, caste flags, state_codes, and apply_url so you can filter deterministically.

Clean legal-aid rules into legal_aid_rules (state_code, income_ceiling, eligible categories, references).

Evaluation:

Parse BhashaBench‑Legal MCQs into bb_legal_questions with question_text, options, correct_option, language, domain, difficulty.

All of this is written back as Silver Delta tables, with schema enforcement and time travel so you can roll back if required.

2.3 Gold: feature-ready for RAG & SQL
Goal: Create fast, query‑optimised views and vector indexes.

In 03_embedding_faiss_indexing.ipynb:

Chunked law text:

Take law_sections, generate smaller law_chunks with overlap (already structured above).

Embeddings:

Use models/embeddings/embedder.py to generate dense vectors for each chunk on CPU (a small sentence‑transformer or similar).

Store as law_chunks_embedded with embedding_en, embedding_hi, etc.

FAISS indexes:

Build legal_index from law_chunks_embedded.

Build scheme_index from embeddings of schemes_curated (name + description + eligibility).

Serialize indexes and store in DBFS/Unity Catalog Volume; paths are registered in core/config.py for easy loading.

Z-ordering:

On schemes_curated, apply Delta Z‑ordering on state_codes, age_min, income_max so scheme filters are extremely fast.

These Gold assets (Z‑ordered tables + FAISS indexes) are what the agents and RAG pipeline actually query at runtime.

3. Where and how the models run (Free Edition + local dev)
3.1 On Databricks Free Edition (for the actual demo)
Constraints: only CPU, ~15 GB RAM, no Databricks Model Serving.

Architecture from your doc:

Use a quantized Param‑1 2.9B model in GGUF Q8_0 (or Q4_K_M) loaded via llama-cpp-python (llama.cpp under the hood).

Memory footprint ~3 GB for Q8_0, leaving RAM for Spark session, FAISS index, and IndicTrans2.

Optionally load a quantized Airavata or Sarvam‑m if you can afford it; but Param‑1 Q8_0 is the safest baseline for the hackathon.

Use IndicTrans2 distilled model (e.g., 1B variant) for translation, also CPU‑only, via CTranslate2 as recommended.

Where they run:

They run inside the same Python process as your app: either the Databricks App container (Streamlit) or an attached notebook cluster.

models/llm/param1_runner.py loads GGUF weights from DBFS into memory at app startup and exposes a generate() function.

agents/translation_agent.py loads IndicTrans2 and exposes to_canonical() / from_canonical().

So at judging time, everything runs inside the Databricks Free Edition workspace – no external APIs or GPUs.

3.2 Local development with the same models
You have two dev loops:

Local machine (your laptop):

Install llama-cpp-python, FAISS CPU, and IndicTrans2 (or at least one translation model).

Keep a subset of data locally (e.g., a few BNS chapters + 20 schemes) to quickly exercise your RAG + agent stack.

Build and test:

models/llm/*.py – model wrappers and quantization logic.

rag/pipeline.py – retrieval + prompt assembly.

agents/*.py – orchestrator / legal / welfare / translation / guardrails.

Databricks sync:

Use Databricks Repo or git integration so nyaya_sahayak/ is the same codebase.

When you commit, pull into Databricks, point model loaders to DBFS paths instead of local filesystem.

Use the same requirements.txt so dependencies match between local and Databricks App.

This way, you explore and debug heavy bits (RAG, prompt logic, agent routing) locally, then run end‑to‑end on Free Edition to ensure you are within memory and latency constraints.

4. Agentic AI in this app: how it actually works
Your agentic pipeline is not just conceptual; it’s a concrete call graph.

4.1 Orchestrator agent – entry point
The orchestrator (agents/orchestrator.py) does:

Receive UserQuery from the UI.

Language handling:

If user language ≠ English/Hindi, call translation_agent.to_canonical() to get a canonical text in English or Hindi and note original language.

Intent classification:

Call models/nlp_classifier/intent_classifier.py to classify into LEGAL, WELFARE, MIXED, or generic.

Routing:

LEGAL → legal_agent.handle(query, persona, act_filter, compare_models, canonical_lang)

WELFARE → welfare_agent.handle(query, user_profile)

MIXED → call both, merge results.

Guardrails:

Optionally pass query to guardrail_agent to block disallowed content, add disclaimers.

Multi‑LLM comparison:

If compare_models=True, call models/llm/comparator.py to generate answers from Param‑1 and (say) Airavata and compute a simple diff.

Translation back:

If original user language ≠ canonical, call translation_agent.from_canonical() on the final answer.

4.2 Legal agent – statutes & IPC→BNS
agents/legal_agent.py:

Uses rag/retriever.py to:

Filter by act_filter if specified (only BNS, only BNSS, etc.) and retrieve top‑k relevant chunks from legal_index.

If query mentions IPC section:

Use ipc_bns_map to look up one or more BNS section numbers and pre‑seed retrieval with those IDs.

Compose a persona‑specific prompt via rag/prompts.py:

Citizen: simpler language, do/don’t guidance, emphasise rights and schemes.

Junior lawyer: more formal, more citations, show IPC→BNS changes.

Call model router: router.select_model(canonical_lang, persona) → Param‑1 (default) or Airavata.

Generate structured LegalAnswer with:

Text, list of section numbers, retrieved context, confidence_score (from vector similarities).

4.3 Welfare agent – schemes & legal-aid
agents/welfare_agent.py:

If user is in wizard flow:

Use deterministic Spark SQL filters on schemes_curated using demographics (age, gender, state, income etc.), then rank results by relevance.

If user typed a question:

Use scheme_index to retrieve relevant schemes and summarise them.

Use legal_aid_rules to check free legal aid eligibility and attach Tele‑Law/Nyaya Bandhu as recommendations.

All of these agents are just Python modules; you can easily test them standalone with unit tests and notebooks.

5. End-to-end request flow (from user to answer)
Putting it all together:

User opens app (Streamlit Databricks App):

Tabs: Citizen Chat, Junior Lawyer Chat, Scheme Wizard, IPC→BNS Explorer, Model Comparison/Eval.

User enters query in one of the chat tabs:

Picks persona, language, optionally act filter and “compare two models” flag.

UI calls orchestrator:

orchestrator.handle(UserQuery)

Orchestrator: translate + classify + route

Maybe calls IndicTrans2 (via translation agent).

Calls intent classifier.

Routes to legal or welfare agent.

Agent runs RAG

RAG uses Gold tables (law_chunks_embedded, schemes_curated) and FAISS indexes for retrieval.

Builds prompt and calls quantized Param‑1 (and optionally a second model for comparison).

LLM(s) generate answer

Answers are combined with citations, scheme recommendations, and confidence scores.

Orchestrator translates back (if necessary) and returns a unified, structured response.

UI renders

Show chat bubbles, list of relevant sections and schemes, toggle to inspect retrieved context, optional side‑by‑side model answers.

6. How to publish the UI app on Databricks
Your doc already outlines the Databricks Apps pattern with Streamlit.

6.1 Code layout for the app
app/main.py – Streamlit entry point:

Imports orchestrator and components.

Creates tabs, session state, and calls orchestrator on submit.

app/components/* – separate files for chat view, scheme wizard, etc.

app.yaml – Databricks App configuration:

Defines Python environment, entry command (e.g., streamlit run app/main.py), resource settings, env vars, secret scopes.

6.2 Steps to deploy as Databricks App
Push repo to GitHub.

In Databricks, create a Databricks App:

Point it to your repo and branch.

Use app.yaml as configuration.

Ensure requirements.txt includes:

streamlit, llama-cpp-python, faiss-cpu, pydantic, sentence-transformers, ctranlsate2 (for IndicTrans2), and Databricks-specific libs.

Deploy via UI or CLI; Databricks Apps will build the environment and host the Streamlit UI under your workspace URL.

This satisfies the “Databricks App / user-facing component” requirement for the hackathon.

7. How you build this in practice (step-by-step dev flow)
Summarising build steps, mixing local and Free Edition:

Local prep (before event)

Quantize Param‑1 to GGUF; verify loading with llama-cpp-python locally.

Prototype RAG pipeline and agentic orchestration on a tiny subset of laws + schemes.

Day 1 – Bronze and Silver

Set up Free Edition workspace + Git integration; upload or programmatically fetch PDFs/CSVs into DBFS.

Run 01_data_ingestion.ipynb → raw Delta tables.

Run 02_data_cleansing_chunking.ipynb → law_sections, schemes_curated, ipc_bns_map, etc.

Day 1 evening – Gold and basic RAG

Run 03_embedding_faiss_indexing.ipynb → embeddings + FAISS indexes.

Implement the simplest rag/pipeline.py and call it from a test notebook.

Overnight – models + agents

Confirm Param‑1 GGUF loads within memory limits on Databricks via param1_runner.py.

Wire up translation_agent, legal_agent, welfare_agent, and orchestrator.

Test E2E for English/Hindi queries via 07_dev_console_ui.ipynb.

Day 2 morning – UI wiring

Implement app/main.py (Streamlit) and components.

Integrate orchestrator into the UI; test all flows.

Run a small BhashaBench-Legal subset via 06_eval_bhashabench.ipynb and log results.

Day 2 afternoon – deployment & polish

Configure app.yaml, deploy Databricks App, and test from a “fresh” browser.

Finalise README and architecture diagram, record 2‑minute demo, rehearse 5‑minute pitch.

