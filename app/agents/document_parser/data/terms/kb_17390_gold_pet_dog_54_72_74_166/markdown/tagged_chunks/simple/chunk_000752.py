from langchain_core.documents import Document

chunk = Document(
    page_content=('- 배기량 또는 정격출력의 크기와 관계없이 1인 또는 2인의 사람을 운송하기에 적합\n'
 '- 하게 제작된 이륜의 자동차 및 그와 유사한 구조로 되어 있는 자동차를 말하며,\n'
 '- 도로교통법(하위 법령, 규칙 포함)에 정한 "원동기장치자전거(전동킥보드, 전동\n'
 '- 이륜평행차, 전동기의 동력만으로 움직일 수 있는 자전거 등 개인형 이동장치를\n'
 '- 포함)"를 포함합니다. 다만, 전동휠체어, 의료용 스쿠터 등 보행보조용 의자차는\n'
 '- 제외합니다.\n'
 '- 용 어 풀 이\n'
 '| 퍼스널모빌리티(세그웨이, | 전동킥보드, 전동이륜평행차 등)는 | 자동차관리법에 |'),
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
 'indexing': {'chunk_id': 'chunk_000752',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
