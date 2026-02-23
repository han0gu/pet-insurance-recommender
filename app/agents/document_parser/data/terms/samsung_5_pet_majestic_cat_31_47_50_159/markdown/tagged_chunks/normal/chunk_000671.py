from langchain_core.documents import Document

chunk = Document(
    page_content=('은 제외합니다)는 자동차관리법에 정한 ‘이륜자동차’, 도로교통법에 정한 ‘원동기장치자전거’에 포\n'
 '함됩니다.③ 제2항에서 “그와 유사한 구조로 되어 있는 자동차”는 다음 각 호에 해당하는 자동차\n'
 '를 포함합니다.- 1. 이륜인 자동차에 측차를 붙인 자동차\n'
 '- 2. 조향장치의 조작방식, 동력전달방식 또는 원동기 냉각방식 등이 이륜의 자동차와\n'
 '유사한 구조로 되어 있는 삼륜 또는 사륜의 자동차로서 승용자동차에 해당하지 않\n'
 '는 자동차3. 전동기를 이용한 동력발생장치를 사용하는 삼륜 또는 사륜의 자동차로서 승용자동'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000671',
              'chunk_char_len': 283,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
