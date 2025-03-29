import pandas as pd
from IPython.display import display


def check_collection_exist(client, name):
    """Try to get a collection; return None if it doesn't exist."""
    try:
        return client.get_collection(name)
    except Exception:
        return None
    

def inspect_chunks(nodes, preview_length: int = 200):
    """Display document chunks with metadata and provide summary statistics.
    
    Args:
        nodes: List of document chunks (nodes) with attributes `text` and `metadata`.
        preview_length: Maximum number of characters to show for the text preview.
    
    Returns:
        A pandas DataFrame containing the chunk details.
    """
    
    chunks_data = []
    for i, node in enumerate(nodes):
        # Safely get text and metadata from node
        text = getattr(node, "text", "")
        metadata = getattr(node, "metadata", {})
        source = metadata.get("file_name", "Unknown")
        preview = text if len(text) <= preview_length else text[:preview_length] + "..."
        
        chunks_data.append({
            "Chunk #": i + 1,
            "Text Preview": preview,
            "Length": len(text),
            "Source": source,
            "Start Idx": metadata.get("start_char_idx", ""),
            "End Idx": metadata.get("end_char_idx", "")
        })
    
    df = pd.DataFrame(chunks_data)
    
    # Calculate summary statistics from nodes
    total_chunks = len(nodes)
    # Unique document sources based on 'Source' column
    total_documents = df["Source"].nunique()
    
    print(f"Total Documents (unique sources): {total_documents}")
    print(f"Total Chunks: {total_chunks}")
    print("\nChunks per Document:")
    print(df["Source"].value_counts())
    print("\nChunk Details:")
    
    # Optionally adjust display options for better readability
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.max_rows", None)
    
    return df





# class EvaluationMetrics:
#     def __init__(self, rag_dataset):
#         """Initialize with RAG dataset and metrics storage"""
#         self.rag_dataset = rag_dataset
#         self.results = []
#         self.retrieval_times: List[float] = []
#         self.retrieval_accuracies: List[float] = []
#         self.semantic_scores: List[float] = []
#         self.mrr_scores: List[float] = []
#         self.ndcg_scores: List[float] = []
#         self.precision_at_k: Dict[int, List[float]] = {1: [], 3: [], 5: []}
#         self.hit_rates: List[float] = []
#         # use global embedding model
#         self.embedding_model = Settings.embed_model

#     def evaluate_retrieval_accuracy(self, retrieved_contexts: List[str], query_idx: int)-> Dict:
#         """
#         Evaluate accuracy of retrieved contexts against reference contexts
        
#         Args:
#             retrieved_contexts: List of retrieved text chunks
#             query_idx: Index of query in dataset
            
#         Returns:
#             Dictionary containing accuracy metrics
#         """
#         reference_contexts = self.rag_dataset.reference_contexts[query_idx]

#         retrieved_texts = [node.node.text for node in retrieved_contexts]
        
#         # Convert contexts to sets of sentences for comparison
#         retrieved_set = set(' '.join(retrieved_texts).split('.'))
#         reference_set = set(' '.join(reference_contexts).split('.'))
        
#         # Calculate metrics
#         correct_retrievals = len(retrieved_set.intersection(reference_set))
#         precision = correct_retrievals / len(retrieved_set) if retrieved_set else 0
#         recall = correct_retrievals / len(reference_set) if reference_set else 0
#         f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

#         mrr = self._calculate_mrr(retrieved_texts, reference_contexts)
#         ndcg = self._calculate_ndcg(retrieved_texts, reference_contexts)
#         # Calculate Hit Rate: 1 if at least one relevant retrieval, else 0.
#         hit = 1.0 if correct_retrievals > 0 else 0.0
#         self.hit_rates.append(hit)

#         # Calculate P@k for different k values
#         precision_k = {}
#         for k in self.precision_at_k.keys():
#             p_at_k = self._calculate_precision_at_k(retrieved_texts, reference_contexts, k)
#             self.precision_at_k[k].append(p_at_k)
#             precision_k[f"p@{k}"] = p_at_k
        
#         # Store scores
#         self.mrr_scores.append(mrr)
#         self.ndcg_scores.append(ndcg)
        
#         return {
#             **precision_k,
#             "mrr": mrr,
#             "ndcg": ndcg,
#             "accuracy": f1,  # Using F1 as an overall accuracy measure
#             "precision": precision,
#             "recall": recall,
#             "hit_rate": hit,
#             "correct_retrievals": correct_retrievals,
#             "total_retrieved": len(retrieved_set),
#             "total_reference": len(reference_set)
#         }
    
#     def _calculate_mrr(self, retrieved, reference) -> float:
#         """Calculate Mean Reciprocal Rank"""
#         for i, doc in enumerate(retrieved, 1):
#             if doc in reference:
#                 return 1.0 / i
#         return 0.0
    
#     def _calculate_ndcg(self, retrieved, reference, k=None) -> float:
#         """Calculate NDCG"""
#         if k is None:
#             k = len(retrieved)
        
#         relevance = [1 if doc in reference else 0 for doc in retrieved[:k]]
#         ideal = sorted(relevance, reverse=True)
        
#         dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance))
#         idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal))
        
#         return dcg / idcg if idcg > 0 else 0.0

#     def _calculate_precision_at_k(self, retrieved, reference, k: int) -> float:
#         """Calculate Precision@K"""
#         retrieved_k = retrieved[:k]
#         relevant_k = sum(1 for doc in retrieved_k if doc in reference)
#         return relevant_k / k if k > 0 else 0.0
        
#     def evaluate_semantic_quality(self, generated_answer: str, query_idx: int) -> Dict:
#         """
#         Evaluate semantic similarity between generated and reference answers
        
#         Args:
#             generated_answer: Generated answer to evaluate
#             query_idx: Index of query in dataset
            
#         Returns:
#             Dictionary containing semantic quality metrics
#         """
#         reference_answer = self.rag_dataset.reference_answers[query_idx]
        
#         # Get embeddings using LlamaIndex's embedding model
#         gen_embedding = self.embedding_model.get_text_embedding(generated_answer)
#         ref_embedding = self.embedding_model.get_text_embedding(reference_answer)
        
#         # Calculate cosine similarity
#         similarity = cosine_similarity(
#             np.array(gen_embedding).reshape(1, -1),
#             np.array(ref_embedding).reshape(1, -1)
#         )[0][0]
        
#         return {
#             "semantic_similarity": similarity,
#             "generated_length": len(generated_answer.split()),
#             "reference_length": len(reference_answer.split())
#         }
    
#     def evaluate_all_queries(self, query_engine, llm=None):
#         """
#         Evaluate all queries in the dataset
        
#         Args:
#             query_engine: RAG query engine for retrieving contexts
#             llm: Language model for generating answers (optional)
#         """
#         print(f"Evaluating {len(self.rag_dataset.queries)} queries...")
        
#         for idx, query in enumerate(tqdm(self.rag_dataset.queries)):
#             # Measure retrieval time and get contexts
#             start_time = time.time()
#             retrieved_contexts = query_engine.retrieve(query)
#             retrieval_time = time.time() - start_time
#             self.retrieval_times.append(retrieval_time)
            
#             # Generate answer if LLM provided
#             generated_answer = ""
#             if llm:
#                 generated_answer = self._generate_answer(query_engine, query)
            
#             # Evaluate retrieval accuracy
#             retrieval_metrics = self.evaluate_retrieval_accuracy(
#                 retrieved_contexts,
#                 idx
#             )
#             self.retrieval_accuracies.append(retrieval_metrics['accuracy'])
            
#             # Evaluate semantic quality if answer generated
#             semantic_score = 0.0
#             if generated_answer:
#                 semantic_metrics = self.evaluate_semantic_quality(
#                     generated_answer,
#                     idx
#                 )
#                 semantic_score = semantic_metrics['semantic_similarity']
#                 self.semantic_scores.append(semantic_score)
            
#             # Store complete results
#             result = {
#                 "query_idx": idx,
#                 "query": query,
#                 "retrieval_time": retrieval_time,
#                 "retrieval_metrics": retrieval_metrics,
#                 "generated_answer": generated_answer,
#                 "semantic_score": semantic_score,
#                 "timestamp": datetime.now().isoformat()
#             }
#             self.results.append(result)
            
#     def get_summary_metrics(self):
#         """Get summary of all evaluation metrics"""
#         summary = {
#             "total_queries": len(self.results),
#             "avg_retrieval_time": np.mean(self.retrieval_times),
#             "avg_retrieval_accuracy": np.mean(self.retrieval_accuracies),
#             "avg_semantic_score": np.mean(self.semantic_scores) if self.semantic_scores else 0.0,
#             "avg_mrr": np.mean(self.mrr_scores),
#             "avg_ndcg": np.mean(self.ndcg_scores),
#             "avg_hit_rate": np.mean(self.hit_rates),
#             "timestamp": datetime.now().isoformat()
#         }
        
#         # Add average P@k scores
#         for k in self.precision_at_k.keys():
#             summary[f"avg_p@{k}"] = np.mean(self.precision_at_k[k])
        
#         return summary
    
#     def plot_results(self):
#         """Enhanced visualization with ranking metrics"""
#         plt.style.use('seaborn-v0_8')
#         fig = plt.figure(figsize=(20, 10))
        
#         # Create grid for subplots
#         gs = fig.add_gridspec(2, 3)
        
#         # Plot 1: Retrieval Times
#         ax1 = fig.add_subplot(gs[0, 0])
#         sns.histplot(self.retrieval_times, kde=True, ax=ax1)
#         ax1.set_title('Retrieval Time Distribution')
#         ax1.set_xlabel('Time (seconds)')
        
#         # Plot 2: Accuracy Metrics
#         ax2 = fig.add_subplot(gs[0, 1])
#         accuracy_data = pd.DataFrame({
#             'F1': self.retrieval_accuracies,
#             'MRR': self.mrr_scores,
#             'NDCG': self.ndcg_scores
#         })
#         sns.boxplot(data=accuracy_data, ax=ax2)
#         ax2.set_title('Ranking Metrics Distribution')
#         ax2.set_ylabel('Score')
        
#         # Plot 3: P@K Values
#         ax3 = fig.add_subplot(gs[0, 2])
#         p_at_k_data = pd.DataFrame({f'P@{k}': scores 
#                                    for k, scores in self.precision_at_k.items()})
#         sns.boxplot(data=p_at_k_data, ax=ax3)
#         ax3.set_title('Precision@K Distribution')
#         ax3.set_ylabel('Score')

#                 # Plot 4: Semantic Scores
#         ax4 = fig.add_subplot(gs[1, 0])
#         if self.semantic_scores:
#             sns.histplot(self.semantic_scores, kde=True, ax=ax4)
#             ax4.set_title('Semantic Score Distribution')
#             ax4.set_xlabel('Semantic Score')
        
#         # Plot 5: Metrics Correlation
#         ax5 = fig.add_subplot(gs[1, 1:])
#         metrics = np.column_stack([
#             self.retrieval_accuracies,
#             self.mrr_scores,
#             self.ndcg_scores
#         ])
#         sns.heatmap(
#             np.corrcoef(metrics.T),
#             annot=True,
#             xticklabels=['F1', 'MRR', 'NDCG'],
#             yticklabels=['F1', 'MRR', 'NDCG'],
#             ax=ax5
#         )
#         ax5.set_title('Metrics Correlation')
        
#         plt.tight_layout()
#         plt.show()
    
#     def save_results(self, filepath):
#         """Save evaluation results with source information"""
#         output = {
#             "summary_metrics": self.get_summary_metrics(),
#             "detailed_results": self.results,
#             "metadata": {
#                 "timestamp": datetime.now().isoformat(),
#                 "total_queries": len(self.results),
#                 "sources": {
#                     "llm": len([r for r in self.results if r.get("source") == "llm"]),
#                     "manual": len([r for r in self.results if r.get("source") == "manual"])
#                 }
#             }
#         }
#         with open(filepath, 'w') as f:
#             json.dump(output, f, indent=2)

#     def _generate_answer(self, query_engine, query):
#         """Helper to generate answer using Query Engine"""
#         try:
#             # Use the query engine directly since it already has the prompt setup
#             response = query_engine.query(query)
#             return str(response)
#         except Exception as e:
#             print(f"Error generating answer: {e}")
#             return ""


# def run_evaluation_pipeline(
#     dataset_path: str,
#     documents,
#     response_prompt: PromptTemplate,
#     use_manual: bool = False,
#     manual_path: str = None,
#     results_path: Path = None
# ) -> tuple:
#     """Run evaluation and optionally save results"""
#     # load llm generated QnA
#     with open(dataset_path, 'r') as f:
#         llm_dataset = json.load(f)
#     # check for manual datasets
#     if use_manual and manual_path:
#         manual_dataset = load_manual_questions(manual_path)

#         # Combine datasets
#         combined_dataset = {
#             "examples": llm_dataset["examples"] + manual_dataset["examples"],
#             "dataset_info": {
#                 "total_questions": len(llm_dataset["examples"]) + len(manual_dataset["examples"]),
#                 "sources": ['llm', 'manual'],
#                 "creation_timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
#             }
#         }
#         eval_dataset = combined_dataset
#     else:
#         eval_dataset = llm_dataset 
        
    
#     # Setup RAG components
#     rag_dataset = RagDataset(eval_dataset)
#     index = VectorStoreIndex.from_documents(documents, show_progress=True)
#     query_engine = index.as_query_engine(
#         response_mode="compact",
#         response_prompt=response_prompt,
#         similarity_top_k=3,
#         max_tokens=300,
#         streaming=False
#     )
    
#     # Run evaluation
#     evaluator = EvaluationMetrics(rag_dataset)
#     evaluator.evaluate_all_queries(query_engine, llm=Settings.llm)
    
#     # Analyze results by source
#     if use_manual:
#         llm_results = [r for r in evaluator.results 
#                       if eval_dataset["examples"][r["query_idx"]]["query_by"]["type"] == "ai"]
#         manual_results = [r for r in evaluator.results 
#                          if eval_dataset["examples"][r["query_idx"]]["query_by"]["type"] == "expert"]
        
#         print("\nResults by Source:")
#         print(f"LLM Questions ({len(llm_results)})")
#         print(f"Manual Questions ({len(manual_results)})")
    
#     return evaluator, eval_dataset