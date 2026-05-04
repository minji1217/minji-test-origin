import re
import nltk 
# from transformers import AutoTokenizer
import config

class QueryBuilder:
    def __init__(self, model_name = config.MODEL_NAME, cite_tag = config.CITE_TAG_PATTERN):
        '''
        문자열 조작 담당하는 클래스 
        온라인 환경에서 기준점이 될 태그 ([CITE]) 설정 
        '''
        self.cite_tag = cite_tag

        # SPECTER2 모델이 읽을 수 있는 형태로 글자 잘라주는 도구 
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name) 

        # [추가] NLTK 문장 토크나이저 다운로드 (최초 1회만 실행)
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('tokenizers/punkt_tab/english/')
        except LookupError:
            print("NLTK punkt 데이터를 다운로드 중...")
            nltk.download('punkt') # 문장 분리 기능에 필요한 지식 데이터베이스 
            nltk.download('punkt_tab')


    def build_offline_query(self, paper_id, full_text, title, abstract, all_references, num_sentences = config.NUM_SENTENCES):
        '''
        [다중 인용 처리 - NLTK 기반 문장 토큰화 적용]
        논문 한 편의 전체 텍스트에서 모든 [CITE] 위치를 찾아 
        각각 (Paper Query, Context Query, Target_Index) 리스트 반환 
        '''
        # 1. paper_id를 인자로 받아 query_id (7단계에서 수행) 생성하여 context_queries에 추가 
        context_queries = []

        # 논문의 전체 레퍼런스 목록을 집합으로 만듦 (-> for 정답 제거)
        all_refs_set = set(all_references if all_references else [])

        paper_query = f"{title} [SEP] {abstract}"

        # 2. 텍스트 내의 모든 [CITE] 위치 찾음
        # 해당 텍스트가 어디서 시작해서 어디에서 끝나는지에 대한 정보 가진 객체 return 
        # 예: [CITE:paper_id_123] 형태라면 r"\[CITE:(.*?)\]" 사용
        matches = list(re.finditer(self.cite_tag, full_text))

        # idx : placeholder idx 
        for idx, match in enumerate(matches):
            start_pos = match.start()

            # 3. GT 추출 
            # match.group(1)은 정규식 (.*?)에 걸린 문자열 (예: "paper_A, paper_B")
            raw_ids = match.group(1) # 1: 정규식 패턴 안의 첫 번째 () 영역에 해당하는 텍스트 

            # 콤마 기준으로 쪼개고, strip 통해 공백 제거하여 리스트 생성 
            # 결과 : ["paper_A", "paper_B"]
            target_ids = [target_id.strip() for target_id in raw_ids.split(",") if target_id.strip()]
            # for -> if -> [strip()]

            # 4. [for 효율성] 해당 인용구 직전의 텍스트 슬라이싱 (placeholder 전 1000글자 정도 1차로 가져옴, 약 150~200단어)
            buffer_zone = max(0, start_pos - 1000)
            raw_snippet = full_text[buffer_zone:start_pos]

            # 5. NLTK 이용해 문장 단위 분할 (리스트로 분해)
            sentences = nltk.sent_tokenize(raw_snippet)
            
            # 5-1. 마지막 3문장만 추출 
            selected_sentences = sentences[-num_sentences:]

            # 6. 첫 문장이 앞부분이 잘려 나간 상태일 경우 온전한 단어부터 시작하도록 
            # if len(sentences) <= num_sentences and len(selected_sentences) > 0:
            #     if " " in selected_sentences[0]:
            #         selected_sentences[0] = selected_sentences[0][selected_sentences[0].find(" ")+1:]

            # 7. 추출된 문장들을 [SEP] 단위로 결합하여 최종 context 구성 
            context_query = " [SEP] ".join(selected_sentences)
            
            
            ''''
            # 5. [for 정밀화] 토큰 단위로 변환 후 뒤에서부터 추출 
            # 토크나이저로 쪼갠 뒤, 정확히 window_size만큼 가져옴
            tokens = self.tokenizer.encode(raw_snippet, add_special_tokens = False) # [CLS], [SEP] 같은 특수 기호 뺌 
            selected_tokens = tokens[-window_size:]

            # 잘라낸 토큰들을 다시 텍스트로 복원 
            context = self.tokenizer.decode(selected_tokens, skip_special_tokens = True)

            # 6. 토큰을 강제로 자르면 깨질 수 있기에 (ex. ##ing) 첫번째 공백을 찾아서 그 이후의 온전한 텍스트부터 가져옴
            # "##ing research paper" -> "research paper"
            if " " in context:
                # 첫번째 공백 위치를 찾고, 그 공백 바로 다음 글자부터 끝까지 다시 가져옴
                # strip 통해 문자열 양쪽 끝에 있는 불필요한 공백 제거 
                context = context[context.find(" ") + 1:].strip() # context[숫자:] : 숫자부터 끝까지 가져옴
            '''
            '''
            # 6. 쿼리 2개에 대해 union
            # Heavy Query (Title + Abstract + Context)
            # Abstract는 512 토큰 초과 방지 위해 앞 150단어, title은 앞 30단어로 제한
            truncated_abstract = " ".join(abstract.split()[:150])
            truncated_title = " ".join(title.split()[:30])
            heavy_query = f"{truncated_title} [SEP] {truncated_abstract} [SEP] {context}"

            # Light Query (Context Only)
            light_query = context

            # 7. 최종 결과 딕셔너리로 묶어 리스트에 추가
            samples.append({
                "heavy_query" : heavy_query,
                "light_query" : light_query,
                "context_pos" : start_pos # [CITE]가 몇 번째 위치였는지 체크 용도 
            })
            '''

            # 7. 쿼리 2개에 대해 union
            # Paper Query (Title + Abstract) -> for문 밖에서 생성 
            # Abstract는 512 토큰 초과 방지 위해 앞 150단어, title은 앞 30단어로 제한
            ''' 
            truncated_abstract = " ".join(abstract.split()[:150])
            truncated_title = " ".join(title.split()[:30])
            '''
    

            # query_id 생성 (예: paper_id = 123, placeholder_idx = 0 -> 123_0 )
            q_id = f"{paper_id}_{idx:02d}"

            # 전체 레퍼런스 집합에서 현재 쿼리 정답 집합을 빼서 누수 방지
            safe_bibs = list(all_refs_set - set(target_ids))

            # 8. 최종 결과 딕셔너리로 묶어 리스트에 추가
            context_queries.append({
                "paper_id" : paper_id,
                "query_id" : q_id,
                "context_query" : context_query,
                "context_pos" : start_pos, # [CITE]가 몇 번째 위치였는지 체크 용도 
                "target_ids" : target_ids, # FAISS 검색 후 채점할 때 쓸 GT 
                "bib_ids" : safe_bibs 
            })

        
        return paper_query, context_queries
    
    def build_online_query(self, user_input_text, title = "", abstract = "", num_sentences = config.NUM_SENTENCES):
        '''
        [실시간 쿼리 빌더] 유저가 \\cite{ 를 입력한 직후 전송된 텍스트(현재 커서 이전 1000글자 정도) 파싱
        '''

        # 1. 유저가 입력한 텍스트에서 \cite{ 의 정확한 위치 찾음 
        raw_snippet = user_input_text.replace("\\cite{", "").strip()

        # NLTK 문장 분리 적용
        sentences = nltk.sent_tokenize(raw_snippet)
        selected_sentences = sentences[-num_sentences:]

        # 문장들을 [SEP] 단위로 결합 
        context_query = " [SEP] ".join(selected_sentences)

        # 4. 최종 쿼리 생성 (paper_query, context_query 생성)
        paper_query = f"{title} [SEP] {abstract}".strip()

        return paper_query, context_query

        


            

