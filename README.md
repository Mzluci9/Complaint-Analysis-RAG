#  Intelligent Complaint Analysis for Financial Services

---

**A Retrieval-Augmented Generation (RAG) chatbot to transform unstructured customer complaints into actionable insights for digital finance teams.**

##  Project Overview

CrediTrust Financial is a fast-growing digital finance platform serving over 500,000 users across East Africa. With thousands of monthly customer complaints submitted via multiple channels, internal teams like Product, Support, and Compliance are overwhelmed and reactive.

This project delivers an **AI-powered internal tool** to streamline complaint analysis and enable real-time decision-making.

##  Business Objective

Build a **RAG-based chatbot** that:

* Lets internal users ask natural-language questions like *"Why are customers unhappy with BNPL?"*
* Returns concise, evidence-backed answers using real complaint narratives
* Helps teams **spot trends in minutes, not days**
* Supports multiple financial products:

  * Credit Cards
  * Personal Loans
  * Buy Now, Pay Later (BNPL)
  * Savings Accounts
  * Money Transfers

## Solution Architecture

1. **Data Preprocessing**

   * Load, clean, and filter complaints from the CFPB dataset
   * Focus on narratives from five key financial products

2. **Text Chunking & Embedding**

   * Split long narratives for better semantic representation
   * Generate embeddings using `all-MiniLM-L6-v2`
   * Index into `FAISS` or `ChromaDB` with metadata

3. **RAG Pipeline**

   * Retrieve top-k similar complaint chunks for each user query
   * Inject into prompt template and generate responses with an LLM
   * Score results and provide source excerpts for trust & transparency

##  KPIs

* Cut down issue detection time from days to minutes
* Enable non-technical teams to get answers independently
* Drive proactive, data-informed product improvements

## Project Structure

```bash
├── data/                    # Cleaned & filtered complaint dataset
├── notebooks/               # EDA and preprocessing notebooks
├── src/                     # Core RAG logic and utility scripts
├── vector_store/            # Persisted embeddings and index
├── app/                     # Streamlit or Gradio interface code
├── report.md                # Summary of EDA, modeling, and evaluation
└── README.md                # Project overview and setup
```

##  Setup Instructions

1. Clone the repo

   ```bash
   git clone https://github.com/your-username/complaint-rag-chatbot.git
   cd complaint-rag-chatbot
   ```

2. Create and activate a virtual environment

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies

   ```bash
   pip install -r requirements.txt
   ```

4. Run the chatbot interface

  ##  Learning Outcomes

* Implemented semantic search using vector databases
* Built a full Retrieval-Augmented Generation pipeline
* Evaluated chatbot responses using qualitative metrics
* Created a transparent, explainable AI system for real-world finance use cases

---



> For any questions or feedback, feel free to raise an issue or reach out via [LinkedIn](https://www.linkedin.com).


