from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 피보험자가 이륜자동차를 직업, 직<br>무 또는 동호회활동과 출퇴근용도 등 주로 사용하게 된 사실을 회사가 입증하지<br>못한 '
 '때에는 보험금을 지급합니다.<br>\uf000 제1항의 이륜자동차라 함은 자동차관리법 시행규칙 제2조에 정한 이륜자동차로 총<br>배기량 '
 '또는 정격출력의 크기와 관계없이 1인 또는 2인의 사람을 운송하기에 적합<br>하게 제작된 이륜의 자동차 및 그와 유사한 구조로 되어 '
 '있는 자동차를 말하며,<br>도로교통법(하위 법령, 규칙 포함)에 정한 "원동기장치자전거(전동킥보드, 전동<br>이륜평행차, 전동기의'),
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
 'indexing': {'chunk_id': 'chunk_001322',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
