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

def process_paper_batch_baseline(paper_batch, query_builder, embedder, retriever, bib_scorer, embedding_db):
    """
    [Baseline 모델 동작 방식]
    인용구(Context) 주변의 텍스트를 무시하고, 오직 논문의 전체 주제(Paper Query)만으로 
    FAISS 검색을 수행하여 모든 인용구 자리에 '동일한 추천 결과' 제공 
    """
    final_output_for_next = []
    
    for item in paper_batch:
        paper_id = item.get('paper_id', '')

        # 1. 쿼리 추출 (여기까지는 동일함)
        paper_query, context_queries = query_builder.build_offline_query(
            paper_id, item.get('full_text',''), item.get('title', ''), item.get('abstract',''), item.get('all_references', [])
        )

        # 2. DB에 존재하는 유효한 문맥만 필터링 (동일함)
        valid_contexts = []
        for sample in context_queries:
            vt = [tid for tid in sample['target_ids'] if tid in embedding_db]
            if vt:
                sample['target_ids'] = vt
                valid_contexts.append(sample)

        if not valid_contexts: continue 

        # =====================================================================
        # 오직 '논문 전체 주제(paper_query)'로만 검색
        # =====================================================================
        p_vec = embedder.encode([paper_query])
        
        # 하이브리드 검색이나 행렬 내적(np.dot)으로 5000개를 뽑아 재계산할 필요 없이,
        # FAISS한테 처음부터 150개(최종 개수)만 깔끔하게 가져옴
        p_res = retriever.search(p_vec, [paper_id], top_k = config.TOP_K_FINAL)[0]

        # =====================================================================
        # [Baseline 핵심 2] 로컬 문맥(c_vecs) 연산 삭제 
        # 원래 기존 코드에 있던 target_matrix 생성, c_vecs 인코딩, np.dot 내적, 
        # 그리고 0.4(p_norm) + 0.6(c_norm) 가중합 로직이 제거됨 
        # =====================================================================

        # 3. 결과 패키징 (각 인용구마다 똑같은 결과를 복사해서 줌)
        for i, sample in enumerate(valid_contexts):
            
            # FAISS가 뽑아준 논문 점수 그대로 후보 리스트 생성
            candidates = []
            for res in p_res:
                candidates.append({
                    "paper_id": res['paper_id'],
                    "sim": float(res['score'])  # FAISS 기본 내적 점수
                })

            # =====================================================================
            # [Baseline 핵심 3] Soft Bias는 유지 (공평한 비교를 위해)
            # 로컬 문맥의 효과'만' 순수하게 비교하기 위해, 
            # 서지 정보(그래프)를 필터링하는 모듈은 똑같이 적용
            # =====================================================================
            raw_bibs = sample.get('bib_ids', [])
            valid_user_bibs = [b for b in raw_bibs if b in embedding_db]
            biased = bib_scorer.soft_bias(candidates, valid_user_bibs, embedding_db)
            
            # 최종 피처 정리 및 정규화
            norm_sims = np.array([c['sim'] for c in biased])
            raw_scores = np.array([c.get('bib_score', 0.0) for c in biased])
            b_min, b_max = np.min(raw_scores), np.max(raw_scores)
            norm_bibs = (raw_scores - b_min) / (b_max - b_min + 1e-9) if b_max > b_min else np.zeros_like(raw_scores)

            clean_candidates = [{
                "paper_id": cand['paper_id'],
                "sim": float(norm_sims[idx]),
                "bib_score": float(norm_bibs[idx])
            } for idx, cand in enumerate(biased)]

            final_output_for_next.append({
                "query_id": sample['query_id'],
                "target_ids": sample['target_ids'],
                "context": sample['context_query'],
                "candidates": clean_candidates
            })

    return final_output_for_next


def run_pipeline(data_path, paper_batch_size):
    print(f"[Baseline(로컬 문맥 무시) 파이프라인 가동 시작...] (데이터: {data_path})")
    start_time = time.time()

    query_builder = QueryBuilder()
    embedder = SpecterEmbedder()
    retriever = FaissRetriever()
    bib_scorer = SoftBiasScorer()

    eval_data = utils.load_json(data_path)
    with open(config.EMBEDDING_DB_PATH, "rb") as f:
        embedding_db = pickle.load(f)

    total_papers = len(eval_data)
    all_processed_queries = [] 

    print(f"총 논문 개수 : {total_papers}개 (논문 {paper_batch_size}개씩 묶어서 처리)")

    total_queries_so_far = 0
    global_metrics = {"Recall@50": 0.0, "Recall@100": 0.0, "Recall@150": 0.0, "MRR": 0.0}

    for i in tqdm(range(0, total_papers, paper_batch_size), desc = "배치 처리중"):
        paper_batch = eval_data[i : i + paper_batch_size]

        # baseline 함수 호출
        batch_results = process_paper_batch_baseline(paper_batch, query_builder, embedder, retriever, bib_scorer, embedding_db)
        
        batch_queries_count = len(batch_results)
        if batch_queries_count > 0:
            batch_metrics = {"Recall@50": 0.0, "Recall@100": 0.0, "Recall@150":0.0, "MRR": 0.0}

            for q_data in batch_results:
                predicted_ids = [cand['paper_id'] for cand in q_data['candidates']]
                gt_ids = q_data['target_ids']
                
                metrics = calculate_metrics(predicted_ids, gt_ids)

                for key in global_metrics:
                    batch_metrics[key] += metrics[key]
                    global_metrics[key] += metrics[key]
            
            total_queries_so_far += batch_queries_count

            print(f"[Baseline Batch 성능] Recall@50: {batch_metrics['Recall@50'] / batch_queries_count:.4f} | Recall@100: {batch_metrics['Recall@100'] / batch_queries_count:.4f} | Recall@150: {batch_metrics['Recall@150'] / batch_queries_count:.4f} | MRR: {batch_metrics['MRR'] / batch_queries_count:.4f}")
        
        all_processed_queries.extend(batch_results)
    
    if total_queries_so_far > 0:
        print("\n" + "="*45)
        print(f"최종 Baseline 전체 성능 (Total Queries: {total_queries_so_far}개)")
        print("="*45)
        for key in global_metrics:
            final_avg = global_metrics[key] / total_queries_so_far
            print(f" - {key}: {final_avg:.4f}")
        print("="*45 + "\n")
   
    print(f"총 소요시간 : {time.time() - start_time: .2f}초")

    return all_processed_queries

if __name__ == "__main__":
    final_data = run_pipeline(config.EVAL_DATA_PATH, config.PAPER_BATCH_SIZE)
    # 저장 파일 이름도 변경
    utils.save_json(final_data, "baseline_offline_output.json") 
    print("'baseline_offline_output.json' 저장 완료")