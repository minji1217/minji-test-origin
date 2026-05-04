import torch
import numpy as np
from transformers import AutoTokenizer
from adapters import AutoAdapterModel
from adapters import Stack
import config

class SpecterEmbedder:
    def __init__(self, model_name = config.MODEL_NAME, adapter_name = config.ADAPTER_NAME):
        '''
        텍스트를 768차원 벡터로 변환하는 클래스로,
        latency 최적화(FP16 양자화)와 인용 예측(Proximity) 어댑터 적용
        '''
        print(f"[{model_name}] 모델 로딩 중...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name) # Adapter2가 읽을 수 있게 단어 단위로 쪼개주는 도구

        # 1. 어댑터 지원 모델로 로드 
        # SPECTER2 엔진은 그대로 가져오되, 어댑터 꽂을 수 있는 기능 활성화하여 가져옴
        self.model = AutoAdapterModel.from_pretrained(model_name)

        # 2. 특정 어댑터(Proximity) 로드 및 활성화 
        # [수정 1] load_adapter가 뱉어내는 '실제 이름(예: PRX)'을 변수에 저장
        self.active_adapter_name = self.model.load_adapter(adapter_name, source="hf")
        
        # [수정 2] PEFT 함수(enable_adapters)는 삭제하고, adapters 전용 함수 사용
        self.model.set_active_adapters(self.active_adapter_name)
        
        # (선택) 확실히 활성화되었는지 확인하는 로그
        print(f"활성화된 어댑터: {self.model.active_adapters}")
        # 3. [Latency 최적화] FP16 연산 적용 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # (GPU일 경우, FP16 적용) 메모리 사용량을 절반으로 줄이고 속도 높임
        if self.device.type == "cuda":
            self.model.to(torch.float16).to(self.device)
        # (CPU일 경우, 기본 정밀도 FP32 실행)
        else:
            self.model.to(self.device)
        
        # 4. 평가 모드 전환 (이미 학습된 Adapter2 가중치 이용)
        self.model.eval()
        print(f"SPECTER2 준비 완료 (사용 장치 : {self.device})")

    def encode(self, texts, batch_size = config.QUERY_BATCH_SIZE):
        '''
        텍스트 리스트 입력받아 FAISS에서 검색 가능한 벡터 리스트 반환
        texts : 문자열 리스트 (예: [paper_query] or [context_query1, context_query2,...])
        batch_size : 한 번에 GPU 올릴 데이터 수 
        '''
        
        # 입력문장이 한 줄이면 리스트로 감싸줌 (아래 for문이 리스트 형식을 기대하기때문에 에러 방지장치)
        # 테스트용으로 바로 한 문장에 대해 임베딩 하는 경우 방지 (예: embedder.encode("이 모델의 최적화 기법은 ..."))
        if isinstance(texts, str): texts = [texts]

        all_embeddings = [] # 임베딩 저장 

        # 5. Batch 단위로 쪼개어 처리 
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            
            # 텍스트를 모델이 읽을 수 있는 텐서(숫자)로 변환 (최대 512 토큰 제한)
            inputs = self.tokenizer(
                batch_texts, # batch 단위로 나눈 텍스트 리스트 (예: 문자열 16개 들어있는 리스트)
                padding=True, # 짧은 문장의 남는 공간은 의미 없는 토큰([PAD])으로 채움 
                truncation=True, 
                max_length=config.MAX_SEQ_LENGTH, # truncation 기준이 되는 최대 토큰 수 
                return_tensors="pt" # 결과물을 파이썬 기본 리스트가 아닌 텐서 형태로 바꿔줌 (파이토치 수학 연산 전용 데이터)
            ).to(self.device) 
            # CPU로 query를 토큰화(글자 -> 숫자 ID)
            # inputs은 행렬 형태의 데이터가 담김 (딕셔너리 안에 들어있는 실제 데이터의 모양이 행렬)
            # to(self.device) 통해 GPU 메모리로 복사 

            # 6. 연산 최적화: 기울기 계산을 끄고 사전학습된 SPECTER2 가중치 사용하여 임베딩 
            with torch.no_grad():
                outputs = self.model(**inputs, adapter_names = [self.active_adapter_name]) # 딕셔너리 자동으로 언패킹 (input_ids, attention_mask)
                
                # 7. 문장 임베딩 추출: SPECTER2는 항상 첫 번째 토큰([CLS])을 해당 문장의 대표 벡터로 사용
                # shape: (batch_size, sequence_length, 768) -> (batch_size, 768)
                embeddings = outputs.last_hidden_state[:, 0, :] # batch_size 문장 다 가져옴 -> 각 문장의 512개 토큰 중 0번째 토큰만 가져옴 -> 그 0번째 토큰의 768차원 모두 가져옴

                # model 통해 연산하는 건 FP16, 그 외엔 FAISS는 FP32 기반이기에 FP32로 변환
                embeddings = embeddings.float()
                
                # 8. L2 정규화 (Normalization)
                # FAISS에서 내적(Inner Product) 연산으로 코사인 유사도를 구하려면 반드시 길이를 1로 맞춰야 함
                normalized_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1) # 길이 1로 L2정규화 통해 변환 
                
                # GPU에 있는 텐서를 CPU로 내리고 Numpy 배열로 변환해서 저장
                all_embeddings.append(normalized_embeddings.cpu().numpy().astype(np.float32))

        # 쪼개진 배치 결과들을 하나의 큰 Numpy 배열로 합쳐서 반환
        # 예: 1round(쿼리1, 쿼리2 : [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]), 2round(쿼리3, 쿼리4: ...)
        '''
        [[0.1, 0.2, 0.3],  # 쿼리 1
        [0.4, 0.5, 0.6],   # 쿼리 2
        [0.7, 0.8, 0.9],   # 쿼리 3
        [1.0, 1.1, 1.2],   # 쿼리 4
        ''' 
        return np.vstack(all_embeddings)