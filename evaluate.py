import os
import glob
import re
import json
import time
import builtins
def print(*args, **kwargs):
    kwargs.setdefault('flush', True)
    builtins.print(*args, **kwargs)

from agent import process_chat, retriever, llm
from langchain_core.messages import HumanMessage

CONVERSATIONS_DIR = r"f:\SHL_AI_Internship_Assignment\sample_conversations\GenAI_SampleConversations"

def extract_user_queries(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract user queries
    queries = re.findall(r"\*\*User\*\*\s*\n+\s*>\s*(.*)", content)
    
    # Extract GT URLs
    gt_urls = re.findall(r"https://www.shl.com/products/product-catalog/view/[\w-]+/?", content)
    # clean trailing slash if any and remove duplicates
    gt_urls = list(set([url.rstrip('/') for url in gt_urls]))
    
    return queries, gt_urls

def evaluate_llm_judge(metric, user_queries, agent_response, context=None):
    if metric == "groundedness":
        prompt = f"Context:\n{context}\n\nAgent Response:\n{agent_response}\n\nRate the groundedness of the agent response based strictly on the context from 1 to 5. Output ONLY the integer 1, 2, 3, 4, or 5."
    elif metric == "effectiveness":
        queries_str = "\n".join([f"- {q}" for q in user_queries])
        prompt = f"User Queries History:\n{queries_str}\n\nAgent Response:\n{agent_response}\n\nRate how effective and accurate the agent response is to the user's latest queries from 1 to 5. Output ONLY the integer 1, 2, 3, 4, or 5."
    
    try:
        res = llm.invoke([HumanMessage(content=prompt)])
        score_match = re.search(r"\b([1-5])\b", res.content)
        if score_match:
            return int(score_match.group(1))
        return 3 # default middle score if parsing fails
    except Exception as e:
        print(f"LLM Judge error: {e}")
        return 3

def main():
    trace_files = glob.glob(os.path.join(CONVERSATIONS_DIR, "*.md"))
    
    results = []
    
    total_recall = 0
    total_relevance = 0
    total_groundedness = 0
    total_effectiveness = 0
    valid_evals = 0
    
    print(f"Found {len(trace_files)} conversation traces. Starting evaluation...")
    
    for filepath in trace_files:
        filename = os.path.basename(filepath)
        queries, gt_urls = extract_user_queries(filepath)
        
        if not queries:
            continue
            
        print(f"\nEvaluating {filename}...")
        
        # Simulate conversation
        messages = []
        final_agent_json = None
        
        for q in queries:
            messages.append({"role": "user", "content": q})
            
            try:
                # call agent
                res = process_chat(messages)
                final_agent_json = res
                messages.append({"role": "assistant", "content": res.get("reply", "")})
                time.sleep(2) # Avoid rate limits
            except Exception as e:
                print(f"Agent error on {filename}: {e}")
                final_agent_json = {"reply": "Error", "recommendations": []}
                break
                
        # Metrics Calculation
        agent_urls = [rec.get("url", "").rstrip('/') for rec in final_agent_json.get("recommendations", [])]
        
        # 1. Recommendation Relevance (Intersection over Union)
        if not gt_urls and not agent_urls:
            relevance = 1.0 # Both correctly made no recommendations
        elif not gt_urls or not agent_urls:
            relevance = 0.0 # One made recommendations, other didn't
        else:
            intersection = set(gt_urls).intersection(set(agent_urls))
            relevance = len(intersection) / len(set(gt_urls).union(set(agent_urls)))
            
        # 2. Retrieval Quality (Recall@K of GT URLs in vector DB)
        # We simulate the last query to get the retrieved docs
        retrieval_recall = 0.0
        context_text = ""
        if gt_urls and retriever:
            try:
                last_query = queries[-1]
                docs = retriever.invoke(last_query)
                retrieved_urls = [doc.metadata.get('url', '').rstrip('/') for doc in docs]
                
                context_text = "\n".join([doc.page_content for doc in docs])
                
                intersection = set(gt_urls).intersection(set(retrieved_urls))
                retrieval_recall = len(intersection) / len(set(gt_urls))
            except Exception as e:
                print(f"Retrieval error: {e}")
                
        # 3. Groundedness
        groundedness = evaluate_llm_judge("groundedness", queries, final_agent_json.get("reply", ""), context=context_text)
        time.sleep(2)
        
        # 4. Effectiveness
        effectiveness = evaluate_llm_judge("effectiveness", queries, final_agent_json.get("reply", ""))
        time.sleep(2)
        
        print(f"  Retrieval Quality (Recall): {retrieval_recall:.2f}")
        print(f"  Recommendation Relevance (IoU): {relevance:.2f}")
        print(f"  Groundedness (1-5): {groundedness}")
        print(f"  Effectiveness (1-5): {effectiveness}")
        
        total_recall += retrieval_recall
        total_relevance += relevance
        total_groundedness += groundedness
        total_effectiveness += effectiveness
        valid_evals += 1
        
        results.append({
            "trace": filename,
            "retrieval_recall": retrieval_recall,
            "relevance_iou": relevance,
            "groundedness": groundedness,
            "effectiveness": effectiveness
        })
        
    if valid_evals > 0:
        print("\n=== FINAL EVALUATION METRICS ===")
        print(f"Average Retrieval Quality (Recall): {total_recall / valid_evals:.2f}")
        print(f"Average Recommendation Relevance (IoU): {total_relevance / valid_evals:.2f}")
        print(f"Average Groundedness (1-5): {total_groundedness / valid_evals:.2f}/5.0")
        print(f"Average Effectiveness (1-5): {total_effectiveness / valid_evals:.2f}/5.0")
        
        with open("evaluation_results.json", "w") as f:
            json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
