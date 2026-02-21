from langchain_core.documents import Document

chunk = Document(
    page_content=('| --- | --- | --- |\n'
 '| 정한 "이륜자동차", 도로교통법에 정한 "원동기장치자전거"에 포함됩니다. \uf000 제2항에서 "그와 유사한 구조로 되어 있는 '
 '자동차"는 다음 각 호에 해당하는 자동 | 정한 "이륜자동차", 도로교통법에 정한 "원동기장치자전거"에 포함됩니다. \uf000 '
 '제2항에서 "그와 유사한 구조로 되어 있는 자동차"는 다음 각 호에 해당하는 자동 | 정한 "이륜자동차", 도로교통법에 정한 '
 '"원동기장치자전거"에 포함됩니다. \uf000 제2항에서 "그와 유사한 구조로 되어 있는 자동차"는 다음 각 호에 해당하는 자동 |'),
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
 'indexing': {'chunk_id': 'chunk_000754',
              'chunk_char_len': 294,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
