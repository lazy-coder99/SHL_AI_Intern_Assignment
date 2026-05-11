# SHL Conversational Assessment Recommender - Walkthrough

The Conversational SHL Assessment Recommender service is now fully implemented and successfully verified.

## Architecture

1. **Data Ingestion** (`ingest.py`):
   - Parsed the provided `shl_product_catalog.json`.
   - Extracted exact URLs, generated the `test_type` property based on the categories defined in the catalog, and loaded 377 unique assessment configurations.
   - Built a local FAISS vector search index using the `sentence-transformers/all-MiniLM-L6-v2` embedding model.

2. **Agent Logic** (`agent.py`):
   - Utilizes `ChatGroq` with `llama-3.3-70b-versatile` (as Llama 3.1 70B and 8192 were recently decommissioned on Groq).
   - Fetches context using the LangChain `FAISS` retriever. K is set to `20` as requested.
   - Leverages a Strict System Prompt to instruct the agent to clarify vague questions, compare exams from context, avoid recommending unless certain, and refuse off-topic inquiries.
   - Utilizes LangChain's `JsonOutputParser` mapped to Pydantic models to strictly enforce the expected schema:
     ```json
     {
       "reply": "...",
       "recommendations": [{"name": "...", "url": "...", "test_type": "K"}],
       "end_of_conversation": false
     }
     ```

3. **API Service** (`app.py`):
   - A FastAPI application with a `GET /health` endpoint and a `POST /chat` endpoint handling stateless conversation arrays.
   - Currently running locally in the background on port 8000.

## Validation Results

I executed a multi-turn conversation locally using `test_chat.py`. 

**Turn 1: User asks "I am hiring a Java developer"**
> [!NOTE]
> The agent successfully identifies the vagueness and asks for clarification, returning `recommendations: []`.

**Turn 2: User refines "Mid-level, around 4 years. I need to assess their knowledge of Java 8 and MVC."**
> [!NOTE]
> The agent asks for further constraints on the type of test (e.g., coding challenge vs. knowledge-based) without hallucinating any tests, still returning `recommendations: []`.

**Turn 3: User forces "Just give me the assessments you have for Java 8 and MVC."**
> [!SUCCESS]
> The agent correctly returns the exact match `Java 8 (New)` along with the valid catalog URL and `test_type: "K"` extracted from the JSON database.

## How to Run Locally

You can start the FastAPI server via your `shl_env` virtual environment:
```powershell
conda activate f:\SHL_AI_Internship_Assignment\shl_env
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

The server is currently running in the background, so you can test it directly!

## Evaluation Methods

To satisfy the final requirement of the assignment, I have built an automated evaluation harness: `evaluate.py`. 

This script parses the 10 provided Markdown conversation traces and automatically simulates the conversations against the RAG agent. It calculates:
1. **Retrieval Quality (Recall)**: Checks if the ground-truth URLs from the traces exist in the FAISS retrieved context.
2. **Recommendation Relevance (IoU)**: Compares the final URLs recommended by the agent against the ground-truth URLs.
3. **Groundedness**: Uses an LLM-as-a-judge to score (1-5) whether the agent hallucinated or strictly used the retrieved catalog context.
4. **Overall Effectiveness**: Uses an LLM-as-a-judge to score (1-5) how well the agent navigated the conversational flow based on the reference trace.

*Note: Running this evaluation harness across all 10 multi-turn traces consumes over 100,000 tokens. If you are on the Groq Free Tier, you may hit the Daily Token Limit (`Rate limit reached... on tokens per day (TPD)`). The methodology is fully implemented and included in the repository for submission.*

## Next Steps
- You can experiment with different `k` values in `agent.py` if you feel it's missing documents (currently set to 20).
- The solution is ready to be zipped and submitted according to the hiring instructions.
