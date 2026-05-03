import faiss
import numpy as np
import utils 
import config 

class FaissRetriever:
    def __init__(self, index_path = config.FAISS_INDEX_PATH, mapping_dict = config.ID_MAPPING_PATH):
        '''
        1. param index_path: 만들어놓은 FAISS 인덱스 파일 경로
        2. param mapping_dict: FAISS의 인덱스 번호(0, 1, 2...)를 실제 논문 ID('mag_123')로 바꿔주는 딕셔너리
        '''
        print(f"FAISS 인덱스 로딩 중... ({index_path})")
        # 1. FAISS 인덱스 로드(.index 파일 불러옴)
        # 이때, FAISS 내적연산(IndexFlatIP)으로 논문 찾아야함 (∵ encode에서 쿼리 임베딩 정규화함)
        self.index = faiss.read_index(index_path) 

        # 2. ID 매핑 딕셔너리 로드 
        # FAISS는 내부적으로 정수 ID밖에 모르기에 실제 논문의 'paper_id'로 바꿔줄 번역기 필요
        print(f"ID 매핑 데이터 로딩 중 ... ({mapping_dict})")
        self.id_mapping = utils.load_pickle(mapping_dict)
        print(f"인덱스 로드 완료 ... (총 {self.index.ntotal}개 논문 존재)")

    def search(self, query_vector, query_ids, top_k, similarity_threshold = config.SIMILARITY_THRESHOLD):
        '''
        [FAISS 기반 nearest neighbor 검색 구현 및 필터링]
        1. param query_vector: embedder.encode()에서 나온 768차원 Numpy 배열 (배치 쿼리일 경우, shape: N, 768)
        2. query_ids : 단일/배치 모두 리스트 형태로, 쿼리 id 담긴 리스트 
        3. source : 각 쿼리가 paper/context query인지 나타냄
        4. param top_k: Top-K 후보 추출 범위 (기본값 75)
        5. param similarity_threshold: 이 점수 미만인 논문은 truncate
        '''
        # 0. FAISS는 2차원(행렬)만 받는데 1차원 배열 받는 경우 처리 (예: 배치 결과 (50, 768) 하나씩 (768,) 쓰는 경우) 
        if len(query_vector.shape) == 1:
            query_vector = query_vector.reshape(1, -1) # indices, distances 행렬 뱉어냄 (한 행마다 하나의 쿼리로, top-k 논문 리스트)

        # 1. [Top-K 추출] FAISS에게 쿼리 벡터를 주고 가장 비슷한 K개를 찾아오라고 명령
        # distances: 유사도 점수 (코사인 유사도), indices: FAISS 내부 번호
        distances, indices = self.index.search(query_vector, top_k)
        
        # 결과를 담을 리스트 (배치 고려)
        total_results = []

        # 쿼리 개수(N)만큼 반복 
        for query_idx in range(len(query_vector)):
            query_id = query_ids[query_idx]
            retrieval_results = []
            rank = 1 # FAISS는 내부에서 유사도 점수 기준으로 내림차순 
        
            # 2. 쿼리별 결과물 순회 
            for score, idx in zip(distances[query_idx], indices[query_idx]):
                # 유사도 점수 리스트, 논문 번호 리스트 -> zip : (0.9, 10) (0.8, 25), ...
                
                # 3. [Similarity Threshold 필터링]
                # FAISS에서 -1이 나오면 검색 결과가 부족하다는 뜻이므로 종료
                # 코사인 유사도가 threshold보다 낮으면 버림
                if idx == -1 or score < similarity_threshold:
                    break


                # 4. 매핑 테이블 통해 FAISS 번호를 실제 논문 ID로 변환
                paper_id = self.id_mapping.get(int(idx), "UNKNOWN_ID") # 사전에 해당 번호 없으면 "UNKNOWN_ID로 대체"

                # 5. [retrieval 결과 포맷 정의]
                retrieval_results.append({
                    "query_id": query_id,
                    "rank": rank,
                    "paper_id": paper_id,
                    "score": round(float(score), 4),
                })

                rank += 1

            total_results.append(retrieval_results)


        # 쿼리가 1개라면 [[쿼리 결과]], 2개 이상 배치로 들어왔다면 [[1번쿼리결과], [2번쿼리결과]...] 형태로 반환
        return total_results