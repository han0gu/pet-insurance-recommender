from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 전동휠체어, 의료용 스쿠터 등 보행보조용 의자차는<br>제외합니다.<br>용 어 풀 이</p><br><table '
 "id='187' "
 "style='font-size:14px'><thead></thead><tbody><tr><td>퍼스널모빌리티(세그웨이,</td><td>전동킥보드, "
 '전동이륜평행차 등)는</td><td>자동차관리법에</td></tr><tr><td colspan="3">정한 "이륜자동차", 도로교통법에 '
 '정한 "원동기장치자전거"에 포함됩니다'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_001324',
              'chunk_char_len': 256,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
