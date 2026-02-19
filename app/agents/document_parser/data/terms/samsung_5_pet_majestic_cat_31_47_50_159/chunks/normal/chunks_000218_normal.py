from langchain_core.documents import Document

chunk = Document(
    page_content=('. 이륜자동차 또는 원동기장치 자전거(전동킥보드, 전동이륜평행차, 전동기의 동력만 으로 움직일 수 있는 자전거 등 개인형 이동장치를 '
 '포함)를 계속적으로 사용(직업, 직무 또는 동호회 활동과 출퇴근용도 등으로 주로 사용하는 경우에 한함)하게 된 경우(다만, 전동휠체어, '
 '의료용 스쿠터 등 보행보조용 의자차는 제외합니다.)'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 54},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000218',
              'chunk_char_len': 178,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
