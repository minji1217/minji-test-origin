"""import numpy as np
import config 
import time 
import pickle
import utils
from query_builder import QueryBuilder
from embedder import SpecterEmbedder
from retriever import FaissRetriever
from soft_bias import SoftBiasScorer
from fusion_var import rank_fusion_var
from evaluate import calculate_metrics
from cascade import cascade_fusion


def process_paper_batch(paper_batch, query_builder, embedder, retriever, bib_scorer, embedding_db, paper_query_top_k = config.PAPER_QUERY_TOP_K):
    # paper_batch : eval_data에서 32개 논문 가져온 리스트 (json 형태)
    # 1. Flatten : 논문 32개 각각의 모든 context를 1차원 리스트로 모음
    paper_query_list = []
    context_query_list = []   # context query 저장 
    metadata_list = []        # 메타데이터 보관소 (QueryBuilder가 만든 딕셔너리 결과물)

    for item in paper_batch:
        paper_id = item.get('paper_id', '')

        # QueryBuilder를 통해 paper query 1개, context query N개 추출 
        paper_query, context_queries = query_builder.build_offline_query(
            paper_id, item.get('full_text',''), item.get('title', ''), item.get('abstract',''), item.get('all_references', [])
        )


        # 해당 논문의 모든 인용구([CITE:])를 context_query_list에 저장
        for sample in context_queries:
            # [for 초기 데이터] db에 존재하는 진짜 정답만 추려냄 
            valid_targets = [tid for tid in sample['target_ids'] if tid in embedding_db]

            # [for 초기 데이터] 
            if not valid_targets: continue 

            # [for 초기 데이터] 
            sample['target_ids'] = valid_targets

            paper_query_list.append(paper_query)
            context_query_list.append(sample['context_query'])
            metadata_list.append(sample) 

    total_samples = len(context_query_list)
    print(f"context 개수: {len(context_query_list)}")

    # 가져온 논문 32개 모두 인용구 하나도 없다면 패스 
    if total_samples == 0: return []

    # 2. 배치 임베딩
    # 2-1. context query 한 번에 임베딩 
    # 이때, embedder 내부에서 batch_size(예: 64) 단위로 쪼개어 연산 후 붙여줌
    p_vectors = embedder.encode(paper_query_list)
    c_vectors = embedder.encode(context_query_list)
    query_ids = [m['query_id'] for m in metadata_list]

    # 2-2. 전역 쿼리는 중복 제거한 paper_batch(예: 32) 개수만큼 인코딩
    # 기존 : {"paper_id : paper_query", " : ", ...}
    # 적용 후 : [[, , ,], [, , ,], ...]

    # 3. 오프라인 필터링 
    # context로 전체 후보 풀 다 뒤지지 않고, paper query로 먼저 FAISS 검색해서 후보 풀 추림 (paper_query_top_k개)
    p_search_results = retriever.search(p_vectors, query_ids, source = ["paper"] * total_samples, top_k = paper_query_top_k)
    
    # 4. 온라인 정밀 타격 (추려진 paper_query_top_k개 안에서만 내적하여 최종 top-100 선발)
    all_fused_results = cascade_fusion(p_search_results, c_vectors, embedding_db)

    final_output_for_next = [] # 다음 단계에 제공

    # 5. Soft Bias 적용 및 최종 피처 패키징 
    for i in range(total_samples):
        meta = metadata_list[i]
        
        candidates = all_fused_results[i]

        # FAISS DB에 존재하는 유저 인용 기록만 남김 
        raw_bibs = meta.get('bib_ids', [])
        valid_user_bibs = [b for b in raw_bibs if b in embedding_db] # db에 존재하는 bib만 남김

        # soft bias 점수 계산
        biased = bib_scorer.soft_bias(candidates, valid_user_bibs, embedding_db)
        # sim, bib_score 정규화
        norm_sims = np.array([c['sim'] for c in biased])
        raw_bibs = np.array([c.get('bib_score', 0.0) for c in biased])
    
        # bib_score 정규화 (Min-Max)
        b_min, b_max = np.min(raw_bibs), np.max(raw_bibs)
        # 만약 bib_score가 전부 0이라서 max=0, min=0인 경우를 대비한 방어 로직
        if b_max == b_min:
            norm_bibs = np.zeros_like(raw_bibs)
        else:
            norm_bibs = (raw_bibs - b_min) / (b_max - b_min + 1e-9)

        # top-100 각 논문에 대해 필요한 피처만 추출
        clean_candidates = []
        for idx, cand in enumerate(biased):
            clean_candidates.append({
                "paper_id": cand['paper_id'],
                # "rrf_score": cand['rrf_score'],
                "sim": float(norm_sims[idx]),
                "bib_score": float(norm_bibs[idx])
            })

        query_packet = {
            "query_id": meta['query_id'],
            'target_ids': meta['target_ids'],
            'context': meta['context_query'],
            'candidates': clean_candidates
        }
        
        final_output_for_next.append(query_packet)

    return final_output_for_next

        





def run_pipeline(data_path, paper_batch_size):
    '''
    [동작 방식] 전체 데이터셋을 논문 단위로 쪼개고, 논문 내에서도 context 단위로 쪼개어 동작
    '''
    print(f"[Offline 실험용 추천 파이프라인 가동 시작...] (데이터: {data_path}")
    start_time = time.time()

    # 1. 모듈 생성 
    query_builder = QueryBuilder()
    embedder = SpecterEmbedder()
    retriever = FaissRetriever()
    bib_scorer = SoftBiasScorer()

    # 2. 데이터셋 로드 (정답지 포함된 JSON 파일)
    eval_data = utils.load_json(data_path)
    with open(config.EMBEDDING_DB_PATH, "rb") as f:
        embedding_db = pickle.load(f)

    total_papers = len(eval_data)
    all_processed_queries = [] # 모든 배치를 1차원으로 통합할 리스트 (할지말지 고민)

    print(f"총 논문 개수 : {total_papers}개 (논문 {paper_batch_size}개씩 묶어서 처리)")

    # 전체 데이터 global metrics 누적 변수 초기화 
    total_queries_so_far = 0
    global_metrics = {"Recall@50": 0.0, "Recall@100": 0.0, "Recall@150": 0.0, "MRR": 0.0}

    # 3. 데이터셋 순회하며 파이프라인 실행 (paper_batch_size(예: 32) 단위로 쪼갬)
    for i in range(0, total_papers, paper_batch_size):
        paper_batch = eval_data[i : i + paper_batch_size]
        # 논문 100개, batch : 32일때 마지막 루프 i=96일땐 96~128(96+32)가 아닌 96~100이어야하므로 min 취함 
        print(f"처리 중 ... 논문 [{i} ~ {min(i + paper_batch_size, total_papers)}] / {total_papers}")

        batch_results = process_paper_batch(paper_batch, query_builder, embedder, retriever, bib_scorer, embedding_db)
        
        # 배치 단위 성능 평가 로직 
        batch_queries_count = len(batch_results)
        if batch_queries_count > 0:
            batch_metrics = {"Recall@50": 0.0, "Recall@100": 0.0, "Recall@150":0.0, "MRR": 0.0}

            for q_data in batch_results:
                predicted_ids = [cand['paper_id'] for cand in q_data['candidates']]
                gt_ids = q_data['target_ids']

                
                # 쿼리당 채점 
                metrics = calculate_metrics(predicted_ids, gt_ids)


                # 배치 및 global metrics에 누적 
                for key in global_metrics:
                    batch_metrics[key] += metrics[key]
                    global_metrics[key] += metrics[key]
            
            total_queries_so_far += batch_queries_count

            # 배치 평균 성능 출력
            print(f"[Batch 성능] Recall@50: {batch_metrics['Recall@50'] / batch_queries_count:.4f} | Recall@100: {batch_metrics['Recall@100'] / batch_queries_count:.4f} | Recall@150: {batch_metrics['Recall@150'] / batch_queries_count:.4f} | MRR: {batch_metrics['MRR'] / batch_queries_count:.4f}")
        
        all_processed_queries.extend(batch_results)# 다음 파트에 합치기 (batch_results 이용할지 말지)
    
    # 모든 배치가 끝난 후 최종 전체 성능 평가 결과 출력
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
"""

"""
original 
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
from fusion_var import rank_fusion_var
from evaluate import calculate_metrics


def process_paper_batch(paper_batch, query_builder, embedder, retriever, bib_scorer, embedding_db, paper_query_top_k = config.PAPER_QUERY_TOP_K):
    # paper_batch : eval_data에서 32개 논문 가져온 리스트 (json 형태)
    final_output_for_next = []
    
    # 논문 단위로 순회 

    for item in paper_batch:
        paper_id = item.get('paper_id', '')

        # QueryBuilder를 통해 paper query 1개, context query N개 추출 
        paper_query, context_queries = query_builder.build_offline_query(
            paper_id, item.get('full_text',''), item.get('title', ''), item.get('abstract',''), item.get('all_references', [])
        )

        # DB에 있는 논문만 GT로 구성, GT가 유효한 문맥만 필터링 
        valid_contexts = []
        # 해당 논문의 모든 인용구([CITE:])를 context_query_list에 저장
        for sample in context_queries:
            # [for 초기 데이터] db에 존재하는 진짜 정답만 추려냄 
            vt = [tid for tid in sample['target_ids'] if tid in embedding_db]
            if vt:
                sample['target_ids'] = vt
                valid_contexts.append(sample)

        if not valid_contexts: continue 

        # 2. 오프라인 필터링 (논문당 1번만 실행)
        # title [SEP] abstract (query_builder에서 처리)
        p_vec = embedder.encode([paper_query])
        p_res = retriever.search(p_vec, [paper_id], top_k = config.PAPER_QUERY_TOP_K)[0]

        # 후보 5000개의 벡터를 DB에서 한꺼번에 추출 
        p_ids = [res['paper_id'] for res in p_res]

        # 이때, embedding_db에 존재하는 논문만 가져와 (i, 벡터)을 담은 리스트 생성 
        # p_ids = [A,B,C,D], embedding db에 A,C만 있어도,
        # valid_data = [(0, vecA), (2,vecC)]로 저장됨 
        # =====================================================================
        # 🚨 [긴급 진단] Stage 1 순수 Recall 자동 측정 (여기부터 복붙!)
        # =====================================================================
        '''
        stage1_hits = 0
        stage1_total = 0
        p_ids_set = set(p_ids) 
        
        for sample in valid_contexts:
            gt_ids = sample['target_ids'] 
            
            # 교집합으로 3000개 안에 들어온 정답 개수 확인
            hits = len(set(gt_ids) & p_ids_set)
            stage1_hits += hits
            stage1_total += len(gt_ids)
            
        if stage1_total > 0:
            stage1_recall = stage1_hits / stage1_total
            # 숫자를 5000으로 박아두지 않고 config 값을 읽어오도록 수정!
            print(f"👉 [진단] {paper_id} 논문의 Stage 1 Recall@{config.PAPER_QUERY_TOP_K}: {stage1_recall:.4f} ({stage1_hits}/{stage1_total})")
        # =====================================================================
        '''
        
        valid_data = [(i, embedding_db[pid]) for i,pid in enumerate(p_ids) if pid in embedding_db]
        if not valid_data: 
            continue
        
        # (1,2,3,...) 
        # ([0.1, 0.2, ...], [0.4, 0.5, ...])
        v_indices, t_vectors = zip(*valid_data)

        # 벡터 -> 행렬 변환 
        target_matrix = np.array(t_vectors).squeeze() # Shape: (5000, 768)
        
        # v_indices에 있는 index 기준으로 p_res에서 score 가져오기 
        valid_p_sims = np.array([p_res[i]['score'] for i in v_indices])
        # 해당 논문 paper id 가져옴 
        valid_p_ids = [p_ids[i] for i in v_indices]

        # 3. 행렬 연산으로 모든 문맥 한꺼번에 계산 
        # [ "deep learning for NLP", "transformer architecture paper","attention mechanism explanation"]
        c_queires = [ctx['context_query'] for ctx in valid_contexts]
        c_vecs = embedder.encode(c_queires) # 문맥들 배치 인코딩 [[0.12, ...], [1,2,...]]

        # 모든 문맥에 대해 한꺼번에 유사도 계산
        # (문맥개수, 5000)
        '''
        c_sims_all =
        [
          [0.91, 0.12, ..., 0.33],   # 문맥1 vs 모든 논문
          [0.44, 0.88, ..., 0.22],   # 문맥2 vs 모든 논문
          ...
        ]
        '''
        c_sims_all = np.dot(c_vecs, target_matrix.T)

        # 4. 문맥별로 최종 순위 계산 및 패키징 
        for i, sample in enumerate(valid_contexts):
            c_sims = c_sims_all[i]
            # 4-1. paper 점수 0~1 정규화 
            p_min, p_max = np.min(valid_p_sims), np.max(valid_p_sims)
            p_norm = (valid_p_sims - p_min) / (p_max - p_min + 1e-8) # 작은 수 더해서 0으로 나누는 경우 방지 


            # 4-2. context 점수 0~1 정규화 
            c_min, c_max = np.min(c_sims), np.max(c_sims)
            c_norm = (c_sims - c_min) / (c_max - c_min + 1e-8) # 반환값 shape: (3000,)

            # 4-3. 가중합도 NumPy로 한 번에 처리
            final_sims = (config.PAPER_SIM_WEIGHT * p_norm) + (config.CONTEXT_SIM_WEIGHT * c_norm)

            # 4-4. top-k 정렬 
            top_idx = np.argsort(final_sims)[::-1][:config.TOP_K_FINAL]

            candidates = []
            
            for rank, idx in enumerate(top_idx):
                candidates.append({
                    "paper_id": valid_p_ids[idx],
                    "sim": float(final_sims[idx])
                })

            # Soft Bias
            raw_bibs = sample.get('bib_ids', [])
            valid_user_bibs = [b for b in raw_bibs if b in embedding_db]
            biased = bib_scorer.soft_bias(candidates, valid_user_bibs, embedding_db)
            
            # 최종 피처 정리
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
    '''
    [동작 방식] 전체 데이터셋을 논문 단위로 쪼개고, 논문 내에서도 context 단위로 쪼개어 동작
    '''
    print(f"[Offline 실험용 추천 파이프라인 가동 시작...] (데이터: {data_path}")
    start_time = time.time()

    # 1. 모듈 생성 
    query_builder = QueryBuilder()
    embedder = SpecterEmbedder()
    retriever = FaissRetriever()
    bib_scorer = SoftBiasScorer()

    # 2. 데이터셋 로드 (정답지 포함된 JSON 파일)
    eval_data = utils.load_json(data_path)
    with open(config.EMBEDDING_DB_PATH, "rb") as f:
        embedding_db = pickle.load(f)

    total_papers = len(eval_data)
    all_processed_queries = [] # 모든 배치를 1차원으로 통합할 리스트 (할지말지 고민)

    print(f"총 논문 개수 : {total_papers}개 (논문 {paper_batch_size}개씩 묶어서 처리)")

    # 전체 데이터 global metrics 누적 변수 초기화 
    total_queries_so_far = 0
    global_metrics = {"Recall@50": 0.0, "Recall@100": 0.0, "Recall@150": 0.0, "MRR": 0.0}

    # 3. 데이터셋 순회하며 파이프라인 실행 (paper_batch_size(예: 32) 단위로 쪼갬)
    for i in tqdm(range(0, total_papers, paper_batch_size), desc = "배치 처리중"):
        paper_batch = eval_data[i : i + paper_batch_size]
        # 논문 100개, batch : 32일때 마지막 루프 i=96일땐 96~128(96+32)가 아닌 96~100이어야하므로 min 취함 
        print(f"처리 중 ... 논문 [{i} ~ {min(i + paper_batch_size, total_papers)}] / {total_papers}")

        batch_results = process_paper_batch(paper_batch, query_builder, embedder, retriever, bib_scorer, embedding_db)
        
        # 배치 단위 성능 평가 로직 
        batch_queries_count = len(batch_results)
        if batch_queries_count > 0:
            batch_metrics = {"Recall@50": 0.0, "Recall@100": 0.0, "Recall@150":0.0, "MRR": 0.0}

            for q_data in batch_results:
                predicted_ids = [cand['paper_id'] for cand in q_data['candidates']]
                gt_ids = q_data['target_ids']

                
                # 쿼리당 채점 
                metrics = calculate_metrics(predicted_ids, gt_ids)


                # 배치 및 global metrics에 누적 
                for key in global_metrics:
                    batch_metrics[key] += metrics[key]
                    global_metrics[key] += metrics[key]
            
            total_queries_so_far += batch_queries_count

            # 배치 평균 성능 출력
            print(f"[Batch 성능] Recall@50: {batch_metrics['Recall@50'] / batch_queries_count:.4f} | Recall@100: {batch_metrics['Recall@100'] / batch_queries_count:.4f} | Recall@150: {batch_metrics['Recall@150'] / batch_queries_count:.4f} | MRR: {batch_metrics['MRR'] / batch_queries_count:.4f}")
        
        all_processed_queries.extend(batch_results)# 다음 파트에 합치기 (batch_results 이용할지 말지)
    
    # 모든 배치가 끝난 후 최종 전체 성능 평가 결과 출력
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
"""

"""
recall@3000 출력용

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
from fusion_var import rank_fusion_var
from evaluate import calculate_metrics

def process_paper_batch(paper_batch, query_builder, embedder, retriever, bib_scorer, embedding_db, paper_query_top_k = config.PAPER_QUERY_TOP_K):
    # paper_batch : eval_data에서 32개 논문 가져온 리스트 (json 형태)
    final_output_for_next = []
    
    # 논문 단위로 순회 
    for item in paper_batch:
        paper_id = item.get('paper_id', '')

        # QueryBuilder를 통해 paper query 1개, context query N개 추출 
        paper_query, context_queries = query_builder.build_offline_query(
            paper_id, item.get('full_text',''), item.get('title', ''), item.get('abstract',''), item.get('all_references', [])
        )

        # DB에 있는 논문만 GT로 구성, GT가 유효한 문맥만 필터링 
        valid_contexts = []
        for sample in context_queries:
            vt = [tid for tid in sample['target_ids'] if tid in embedding_db]
            if vt:
                sample['target_ids'] = vt
                valid_contexts.append(sample)

        if not valid_contexts: continue 

        # 2. 오프라인 필터링 (논문당 1번만 실행)
        p_vec = embedder.encode([paper_query])
        p_res = retriever.search(p_vec, [paper_id], top_k = config.PAPER_QUERY_TOP_K)[0]

        # 후보 5000개의 벡터를 DB에서 한꺼번에 추출 
        p_ids = [res['paper_id'] for res in p_res]

        # ✨ [수정 1] 평가를 위해 교집합용 집합(set) 미리 생성
        p_ids_set = set(p_ids) 
        
        valid_data = [(i, embedding_db[pid]) for i,pid in enumerate(p_ids) if pid in embedding_db]
        if not valid_data: 
            continue
        
        v_indices, t_vectors = zip(*valid_data)

        # 벡터 -> 행렬 변환 
        target_matrix = np.array(t_vectors).squeeze() # Shape: (5000, 768)
        
        # v_indices에 있는 index 기준으로 p_res에서 score 가져오기 
        valid_p_sims = np.array([p_res[i]['score'] for i in v_indices])
        valid_p_ids = [p_ids[i] for i in v_indices]

        # 3. 행렬 연산으로 모든 문맥 한꺼번에 계산 
        c_queires = [ctx['context_query'] for ctx in valid_contexts]
        c_vecs = embedder.encode(c_queires) 

        c_sims_all = np.dot(c_vecs, target_matrix.T)

        # 4. 문맥별로 최종 순위 계산 및 패키징 
        for i, sample in enumerate(valid_contexts):
            c_sims = c_sims_all[i]
            
            # 4-1. paper 점수 0~1 정규화 
            p_min, p_max = np.min(valid_p_sims), np.max(valid_p_sims)
            p_norm = (valid_p_sims - p_min) / (p_max - p_min + 1e-8) 

            # 4-2. context 점수 0~1 정규화 
            c_min, c_max = np.min(c_sims), np.max(c_sims)
            c_norm = (c_sims - c_min) / (c_max - c_min + 1e-8)

            # 4-3. 가중합도 NumPy로 한 번에 처리
            final_sims = (config.PAPER_SIM_WEIGHT * p_norm) + (config.CONTEXT_SIM_WEIGHT * c_norm)

            # 4-4. top-k 정렬 
            top_idx = np.argsort(final_sims)[::-1][:config.TOP_K_FINAL]

            candidates = []
            for rank, idx in enumerate(top_idx):
                candidates.append({
                    "paper_id": valid_p_ids[idx],
                    "sim": float(final_sims[idx])
                })

            # Soft Bias
            raw_bibs = sample.get('bib_ids', [])
            valid_user_bibs = [b for b in raw_bibs if b in embedding_db]
            biased = bib_scorer.soft_bias(candidates, valid_user_bibs, embedding_db)
            
            norm_sims = np.array([c['sim'] for c in biased])
            raw_scores = np.array([c.get('bib_score', 0.0) for c in biased])
            b_min, b_max = np.min(raw_scores), np.max(raw_scores)
            norm_bibs = (raw_scores - b_min) / (b_max - b_min + 1e-9) if b_max > b_min else np.zeros_like(raw_scores)

            clean_candidates = [{
                "paper_id": cand['paper_id'],
                "sim": float(norm_sims[idx]),
                "bib_score": float(norm_bibs[idx])
            } for idx, cand in enumerate(biased)]

            # ✨ [수정 2] 해당 문맥 쿼리의 Stage 1 정답률 계산
            stage1_hits = len(set(sample['target_ids']) & p_ids_set)
            stage1_total = len(sample['target_ids'])

            final_output_for_next.append({
                "query_id": sample['query_id'],
                "target_ids": sample['target_ids'],
                "context": sample['context_query'],
                "candidates": clean_candidates,
                "stage1_hits": stage1_hits,       # 👈 메인으로 넘길 데이터 1
                "stage1_total": stage1_total      # 👈 메인으로 넘길 데이터 2
            })

    return final_output_for_next
"""

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

def softmax_norm(x, temp=0.05):
    x = x - np.max(x)
    exp_x = np.exp(x / temp)
    return exp_x / (np.sum(exp_x) + 1e-9)

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

def softmax_norm(x, temp=0.05):
    x = x - np.max(x)
    exp_x = np.exp(x / temp)
    return exp_x / (np.sum(exp_x) + 1e-9)
'''
def rrf_fusion(result_lists, k=config.RRF_K):

    rrf_scores = {}

    for res_list in result_lists:

        for rank, item in enumerate(res_list):

            pid = item["paper_id"]

            score = 1.0 / (k + rank + 1)

            rrf_scores[pid] = (
                rrf_scores.get(pid, 0.0)
                + score
            )

    return rrf_scores
'''
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


        # 3. 행렬 연산으로 모든 문맥 한꺼번에 계산 
        c_queires = [ctx['context_query'] for ctx in valid_contexts]
        c_vecs = embedder.encode(c_queires) 

        c_sims_all = np.dot(c_vecs, target_matrix.T)

        # 4. 문맥별로 최종 순위 계산 및 패키징 
        '''
        for i, sample in enumerate(valid_contexts):
            c_sims = c_sims_all[i]
            
            
            #p_min, p_max = np.min(valid_p_sims), np.max(valid_p_sims)
            #p_norm = (valid_p_sims - p_min) / (p_max - p_min + 1e-8)
#
            #c_min, c_max = np.min(c_sims), np.max(c_sims)
            #c_norm = (c_sims - c_min) / (c_max - c_min + 1e-8)


            #paper_w, context_w = compute_dynamic_weights(c_sims)
            
            final_sims = (
                 (valid_p_sims ** config.PAPER_SIM_WEIGHT) 
                *
                (c_sims ** config.CONTEXT_SIM_WEIGHT)  
            )

            top_idx = np.argsort(final_sims)[::-1][:config.TOP_K_FINAL]

            candidates = []
            for rank, idx in enumerate(top_idx):
                candidates.append({
                    "paper_id": valid_p_ids[idx],
                    "sim": float(final_sims[idx])
                })

            raw_bibs = sample.get('bib_ids', [])
            valid_user_bibs = [b for b in raw_bibs if b in embedding_db]
            biased = bib_scorer.soft_bias(candidates, valid_user_bibs, embedding_db)
            
            norm_sims = np.array([c['sim'] for c in biased])
            raw_scores = np.array([c.get('bib_score', 0.0) for c in biased])
            b_min, b_max = np.min(raw_scores), np.max(raw_scores)
            norm_bibs = (raw_scores - b_min) / (b_max - b_min + 1e-9) if b_max > b_min else np.zeros_like(raw_scores)

            clean_candidates = [{
                "paper_id": cand['paper_id'],
                "sim": float(norm_sims[idx]),
                "bib_score": float(norm_bibs[idx])
            } for idx, cand in enumerate(biased)]

            # ✨ 합집합 풀(union_pool_set) 안에 정답이 있는지 채점
            stage1_hits = len(set(sample['target_ids']) & union_pool_set)
            stage1_total = len(sample['target_ids'])

            final_output_for_next.append({
                "query_id": sample['query_id'],
                "target_ids": sample['target_ids'],
                "context": sample['context_query'],
                "candidates": clean_candidates,
                "stage1_hits": stage1_hits,      
                "stage1_total": stage1_total      
            })'''
        # 4. 문맥별로 최종 순위 계산 및 패키징 
        for i, sample in enumerate(valid_contexts):
            c_sims = c_sims_all[i]
            
            # =====================================================================
            # ✨ [STEP 1] 텍스트 기하평균 (더하기 '+' 절대 금지! 반드시 곱하기 '*' 사용)
            # =====================================================================
            text_sims = (valid_p_sims ** config.PAPER_SIM_WEIGHT) * (c_sims ** config.CONTEXT_SIM_WEIGHT)
            #text_sims = (valid_p_sims * config.PAPER_SIM_WEIGHT) + (c_sims * config.CONTEXT_SIM_WEIGHT)
           # =====================================================================
            # 🚀 [STEP 1.5] 속도 최적화: 텍스트 상위 500명만 먼저 추려내기! 
            # 어차피 500등 밖은 20% 보너스 받아도 150등 안에 못 들어옴
            # =====================================================================
            BIB_CANDIDATE_SIZE = config.BIB_CANDIDATE_SIZE # config.py로 빼도 좋아! (500~800 추천)
            
            if len(text_sims) > BIB_CANDIDATE_SIZE:
                top_text_idx = np.argsort(text_sims)[::-1][:BIB_CANDIDATE_SIZE]
            else:
                top_text_idx = np.arange(len(text_sims))
                
            # 딱 500명 정보만 발췌
            surviving_text_sims = text_sims[top_text_idx]
            surviving_p_ids_for_bib = [valid_p_ids[idx] for idx in top_text_idx]

            # =====================================================================
            # ✨ [STEP 2] 압축된 500명에게만 Bib Score 연산! (연산량 90% 감소)
            # =====================================================================
            raw_bibs = sample.get('bib_ids', [])
            valid_user_bibs = [b for b in raw_bibs if b in embedding_db]
            
            temp_candidates = [{"paper_id": pid} for pid in surviving_p_ids_for_bib]
            biased_all = bib_scorer.soft_bias(temp_candidates, valid_user_bibs, embedding_db)
            
            raw_bib_scores = np.array([c.get('bib_score', 0.0) for c in biased_all])
            b_min, b_max = np.min(raw_bib_scores), np.max(raw_bib_scores)
            norm_bibs = (raw_bib_scores - b_min) / (b_max - b_min + 1e-9) if b_max > b_min else np.zeros_like(raw_bib_scores)

            # =====================================================================
            # ✨ [STEP 3 & 4] 최종 증폭 및 150명 칼질!
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

            # ✨ 합집합 풀(union_pool_set) 안에 정답이 있는지 채점
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

# ... (아래 run_pipeline과 __main__ 부분은 기존과 동일하므로 생략 없이 그대로 쓰면 됩니다!) ...

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
    # ✨ [수정 3] Stage1_Recall 전광판에 추가
    global_metrics = {"Stage1_Recall": 0.0, "Recall@50": 0.0, "Recall@100": 0.0, "Recall@150": 0.0, "MRR": 0.0}

    for i in tqdm(range(0, total_papers, paper_batch_size), desc = "배치 처리중"):
        paper_batch = eval_data[i : i + paper_batch_size]
        print(f"처리 중 ... 논문 [{i} ~ {min(i + paper_batch_size, total_papers)}] / {total_papers}")

        batch_results = process_paper_batch(paper_batch, query_builder, embedder, retriever, bib_scorer, embedding_db)
        
        batch_queries_count = len(batch_results)
        if batch_queries_count > 0:
            # ✨ [수정 4] 배치 전광판에도 동일하게 추가
            batch_metrics = {"Stage1_Recall": 0.0, "Recall@50": 0.0, "Recall@100": 0.0, "Recall@150":0.0, "MRR": 0.0}

            for q_data in batch_results:
                predicted_ids = [cand['paper_id'] for cand in q_data['candidates']]
                gt_ids = q_data['target_ids']
                
                metrics = calculate_metrics(predicted_ids, gt_ids)

                # ✨ [수정 5] 쿼리 1개 단위로 Stage 1 방어율 채점해서 metrics에 합치기
                s1_hits = q_data.get('stage1_hits', 0)
                s1_total = q_data.get('stage1_total', 0)
                metrics["Stage1_Recall"] = s1_hits / s1_total if s1_total > 0 else 0.0

                for key in global_metrics:
                    batch_metrics[key] += metrics[key]
                    global_metrics[key] += metrics[key]
            
            total_queries_so_far += batch_queries_count

            # ✨ [수정 6] 프린트문에 Stage1_Recall 출력 추가
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
