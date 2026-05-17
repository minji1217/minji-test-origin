import numpy as np
import config 
import time 
import pickle
import utils
from tqdm import tqdm 
from query_builder import QueryBuilder
from embedder import SpecterEmbedder
from retriever import FaissRetriever
from soft_bias import SoftBiasScorer
from evaluate import calculate_metrics


def process_paper_batch(paper_batch, query_builder, embedder, retriever, bib_scorer, embedding_db):
    final_output_for_next = []
    
    for item in paper_batch:
        paper_id = item.get('paper_id', '')

        paper_query, context_queries = query_builder.build_offline_query(
            paper_id, item.get('full_text',''), item.get('title', ''), item.get('abstract',''), item.get('all_references', [])
        )

        valid_contexts = []
        for sample in context_queries:
            vt = [tid for tid in sample['target_ids'] if tid in embedding_db]
            if vt:
                sample['target_ids'] = vt
                valid_contexts.append(sample)

        if not valid_contexts: continue 

        # 1. 오직 Full Query 1개만 벡터로 변환
        p_vec = embedder.encode([paper_query])[0] # Shape: (768,)
        
        # 2. FAISS 단일 검색 (합집합 로직 다 버리고 그냥 FULL_TOPK개 가져옴)
        full_res = retriever.search(p_vec, [paper_id], top_k=config.FULL_TOPK)[0]
        
        # 3. 단일 후보 풀(Pool) 생성
        p_ids = [r["paper_id"] for r in full_res]
        union_pool_set = set(p_ids) # Stage 1 Recall 채점용

        # 4. DB에 있는 유효한 임베딩만 걸러내기
        valid_data = [(i, embedding_db[pid]) for i,pid in enumerate(p_ids) if pid in embedding_db]
        if not valid_data: 
            continue
        
        v_indices, t_vectors = zip(*valid_data)
        target_matrix = np.array(t_vectors).squeeze() # Shape: (후보 개수, 768)
        valid_p_ids = [p_ids[i] for i in v_indices]

        # 5. 가중합 짬뽕 대신, 순수하게 Full Query와의 내적 점수 하나만 사용!
        valid_p_sims = np.dot(p_vec, target_matrix.T).squeeze()


        # 6. 행렬 연산으로 모든 문맥 한꺼번에 계산 
        c_queires = [ctx['context_query'] for ctx in valid_contexts]
        c_vecs = embedder.encode(c_queires) 

        c_sims_all = np.dot(c_vecs, target_matrix.T)

        
        # 7. 문맥별로 최종 순위 계산 및 패키징 
        for i, sample in enumerate(valid_contexts):
            c_sims = c_sims_all[i]
            
            # =====================================================================
            # [STEP 1] 가중합
            # =====================================================================
            #text_sims = (valid_p_sims ** config.PAPER_SIM_WEIGHT) * (c_sims ** config.CONTEXT_SIM_WEIGHT)
            text_sims = (valid_p_sims * config.PAPER_SIM_WEIGHT) + (c_sims * config.CONTEXT_SIM_WEIGHT)
            # =====================================================================
            # [STEP 1.5] 속도 최적화: 텍스트 상위 1500명만 먼저 추려낸다
            # =====================================================================
            BIB_CANDIDATE_SIZE = config.BIB_CANDIDATE_SIZE # config.py로 빼도 좋아! (500~800 추천)
            
            if len(text_sims) > BIB_CANDIDATE_SIZE:
                top_text_idx = np.argsort(text_sims)[::-1][:BIB_CANDIDATE_SIZE]
            else:
                top_text_idx = np.arange(len(text_sims))
                
            # 딱 1500명 정보만 GET
            surviving_text_sims = text_sims[top_text_idx]
            surviving_p_ids_for_bib = [valid_p_ids[idx] for idx in top_text_idx]

            # =====================================================================
            # [STEP 2] 압축된 1500명에게만 Bib Score 연산
            # =====================================================================
            raw_bibs = sample.get('bib_ids', [])
            valid_user_bibs = [b for b in raw_bibs if b in embedding_db]
            
            temp_candidates = [{"paper_id": pid} for pid in surviving_p_ids_for_bib]
            biased_all = bib_scorer.soft_bias(temp_candidates, valid_user_bibs, embedding_db)
            
            raw_bib_scores = np.array([c.get('bib_score', 0.0) for c in biased_all])
            b_min, b_max = np.min(raw_bib_scores), np.max(raw_bib_scores)
            norm_bibs = (raw_bib_scores - b_min) / (b_max - b_min + 1e-9) if b_max > b_min else np.zeros_like(raw_bib_scores)

            # =====================================================================
            # [STEP 3 & 4] 최종 증폭 및 top-150 선정
            # =====================================================================
            bib_weight = config.BIB_WEIGHT 
            final_sims = surviving_text_sims + (bib_weight * norm_bibs)

            final_top_idx = np.argsort(final_sims)[::-1][:config.TOP_K_FINAL]

            clean_candidates = []
            for rank, local_idx in enumerate(final_top_idx):
                clean_candidates.append({
                    "paper_id": surviving_p_ids_for_bib[local_idx],
                    "sim": float(surviving_text_sims[local_idx]), 
                    "bib_score": float(norm_bibs[local_idx]) 
                })

            # 합집합 풀(union_pool_set) 안에 정답이 있는지 채점
            stage1_hits = len(set(sample['target_ids']) & union_pool_set)
            stage1_total = len(sample['target_ids'])

            final_output_for_next.append({
                "query_id": sample['query_id'],
                "target_ids": sample['target_ids'],
                "context": sample['context_query'],
                "candidates": clean_candidates,
                "stage1_hits": stage1_hits,      
                "stage1_total": stage1_total      
            })

    return final_output_for_next

def run_pipeline(data_path, paper_batch_size):
    print(f"new branch1 [Offline 실험용 추천 파이프라인 가동 시작...] (데이터: {data_path})")
    start_time = time.time()

    # 1. 모듈 생성 
    query_builder = QueryBuilder()
    embedder = SpecterEmbedder()
    retriever = FaissRetriever()
    bib_scorer = SoftBiasScorer()

    # 2. 데이터셋 로드 
    eval_data = utils.load_json(data_path)
    with open(config.EMBEDDING_DB_PATH, "rb") as f:
        embedding_db = pickle.load(f)

    total_papers = len(eval_data)
    all_processed_queries = [] 

    print(f"총 논문 개수 : {total_papers}개 (논문 {paper_batch_size}개씩 묶어서 처리)")

    total_queries_so_far = 0
    # Stage1_Recall 전광판에 추가
    global_metrics = {"Stage1_Recall": 0.0, "Recall@50": 0.0, "Recall@100": 0.0, "Recall@150": 0.0, "MRR": 0.0}

    for i in tqdm(range(0, total_papers, paper_batch_size), desc = "배치 처리중"):
        paper_batch = eval_data[i : i + paper_batch_size]
        print(f"처리 중 ... 논문 [{i} ~ {min(i + paper_batch_size, total_papers)}] / {total_papers}")

        batch_results = process_paper_batch(paper_batch, query_builder, embedder, retriever, bib_scorer, embedding_db)
        
        batch_queries_count = len(batch_results)
        if batch_queries_count > 0:
            # 배치에도 동일하게 추가
            batch_metrics = {"Stage1_Recall": 0.0, "Recall@50": 0.0, "Recall@100": 0.0, "Recall@150":0.0, "MRR": 0.0}

            for q_data in batch_results:
                predicted_ids = [cand['paper_id'] for cand in q_data['candidates']]
                gt_ids = q_data['target_ids']
                
                metrics = calculate_metrics(predicted_ids, gt_ids)

                # 쿼리 1개 단위로 Stage 1 방어율 채점해서 metrics에 합치기
                s1_hits = q_data.get('stage1_hits', 0)
                s1_total = q_data.get('stage1_total', 0)
                metrics["Stage1_Recall"] = s1_hits / s1_total if s1_total > 0 else 0.0

                for key in global_metrics:
                    batch_metrics[key] += metrics[key]
                    global_metrics[key] += metrics[key]
            
            total_queries_so_far += batch_queries_count

            # Stage1_Recall 출력 추가
            print(f"[Batch 성능] Stage1_Recall: {batch_metrics['Stage1_Recall'] / batch_queries_count:.4f} | Recall@50: {batch_metrics['Recall@50'] / batch_queries_count:.4f} | Recall@100: {batch_metrics['Recall@100'] / batch_queries_count:.4f} | Recall@150: {batch_metrics['Recall@150'] / batch_queries_count:.4f} | MRR: {batch_metrics['MRR'] / batch_queries_count:.4f}")
        
        all_processed_queries.extend(batch_results)
    
    if total_queries_so_far > 0:
        print("\n" + "="*45)
        print(f"최종 전체 성능 (Total Queries: {total_queries_so_far}개)")
        print("="*45)
        for key in global_metrics:
            final_avg = global_metrics[key] / total_queries_so_far
            print(f" - {key}: {final_avg:.4f}")
        print("="*45 + "\n")
   
    print(f"총 소요시간 : {time.time() - start_time: .2f}초")

    return all_processed_queries

if __name__ == "__main__":
    final_data = run_pipeline(config.EVAL_DATA_PATH, config.PAPER_BATCH_SIZE)
    utils.save_json(final_data, "offline_output.json") 
    print("'offline_output.json' 저장 완료")
