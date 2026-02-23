from langchain_core.documents import Document

chunk = Document(
    page_content=('법령, 규칙 포함)에 정한 "원동기장치자전거(전동킥보드, 전동<br>이륜평행차, 전동기의 동력만으로 움직일 수 있는 자전거 등 개인형 '
 '이동장치를<br>포함)"를 포함합니다'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_001323',
              'chunk_char_len': 95,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
