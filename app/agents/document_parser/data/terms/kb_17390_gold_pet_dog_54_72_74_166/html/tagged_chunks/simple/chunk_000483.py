from langchain_core.documents import Document

chunk = Document(
    page_content=('않습니다.<br>\uf000 피보험자가 병원 또는 의원을 이전하여 입원한 경우에도 동일한 상해의 치료를<br>목적으로 2회 이상 입원한 '
 '경우에는 계속하여 입원한 것으로 보아 각 입원일수를<br>더합니다.<br>\uf000 보험수익자와 회사가 제1조(보험금의 지급사유)의 '
 '보험금 지급사유에 대해 합의<br>하지 못할 때는 보험수익자와 회사가 함께 제3자를 정하고 그 제3자의 의견에 따<br>를 수 있습니다'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000483',
              'chunk_char_len': 216,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
