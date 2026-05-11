import os # 컴퓨터 파일이나 폴더 다룸 
# 1. 경로 설정 
'''
__file__ : 현재 실행 중인 이 파이썬 파일 자체 
abspath 통해 파일의 전체 주소를 절대 경로로 바꿔줌
os.path.dirname :  전체 주소에서 파일 이름은 빼고, 그 파일이 들어있는 폴더 주소만 가져옴
'''
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # 기본 폴더 작성 
DATA_DIR = os.path.join(BASE_DIR, "data")

EMBEDDING_DB_PATH = os.path.join(DATA_DIR, "embedding_db.pkl") # 논문id-SPECTER2 임베딩 매핑 저장 파일 경로 
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "candidates.index") # 실제 FAISS 인덱스 저장 파일 경로
ID_MAPPING_PATH = os.path.join(DATA_DIR, "id_mapping.pkl") # FAISS 인덱스-논문 매핑 저장 파일 경로 


# 테스트용(온라인)
# 더미 데이터 경로를 BASE_DIR과 합쳐서 '절대 경로'로 못 박아버림
# FAISS_INDEX_PATH = os.path.join(DATA_DIR, "dummy_faiss.index")
# ID_MAPPING_PATH = os.path.join(DATA_DIR, "dummy_mapping.pkl")
# EMBEDDING_DB_PATH = os.path.join(DATA_DIR, "dummy_embed_db.pkl")

# 테스트용(오프라인)
# FAISS_INDEX_PATH = "dummy_faiss.index"
# ID_MAPPING_PATH = "dummy_mapping.pkl"
# EMBEDDING_DB_PATH = "dummy_embed_db.pkl"
# EVAL_DATA_PATH = "test_eval_data.json" # 이 파일이 배치처리됨

# 1-1. 데이터 경로 

EVAL_DATA_PATH = os.path.join(DATA_DIR, "eval_dataset.json") # 튜닝 위한 val 데이터셋
# TEST_DATA_PATH = os.path.join(DATA_DIR, "test_dataset.json") # 최종 평가용 test 데이터셋 

# 2. 모델/정규식
MODEL_NAME = "allenai/specter2_base"
ADAPTER_NAME = "allenai/specter2_proximity"
CITE_TAG_PATTERN = r"\[CITE:(.*?)\]"

# 3. Retrieval & Fusion 하이퍼파라미터 설정
NUM_SENTENCES = 3           # Context Query 생성시 placeholder 기준 자를 문장 수 
SIMILARITY_THRESHOLD = 0.0  # FAISS 코사인 유사도 최소 임계값
FULL_TOPK = 4000    # paper query 기준 추려올 후보 논문 개수 
TITLE_TOPK = 500
ABSTRACT_TOPK = 500
FULL_WT = 0.75
TITLE_WT = 0.15
ABSTRACT_WT = 0.1
BIB_WEIGHT = 0.2
TOP_K_FINAL = 150           # 후보 논문들 중 context query 통해 가져올 후보 개수 
# RRF_K = 60                # RRF 스무딩 상수 
PAPER_BATCH_SIZE = 256      # 논문 배치 크기 (for main)
QUERY_BATCH_SIZE = 256      # 쿼리 배치 크기 (for encode) 
MAX_SEQ_LENGTH = 512        # SPECTER2 최대 입력 크기 
PAPER_SIM_WEIGHT = 0.35     # 가중합 비율 (paper_query) 
CONTEXT_SIM_WEIGHT = 0.65    # 가중합 비율 (context_query)
