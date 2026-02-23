from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 장해분류표의 각 장해분류별 최저 지급률 장해정도에 이르지 않<br>는 후유장해에 대하여는 후유장해보험금을 지급하지 '
 '않습니다.<br>\uf000 보험수익자와 회사가 제1조(보험금의 지급사유)의 보험금 지급사유에 대해 합의<br>하지 못할 때는 '
 '보험수익자와 회사가 함께 제3자를 정하고 그 제3자의 의견에 따<br>를 수 있습니다'),
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
 'indexing': {'chunk_id': 'chunk_000340',
              'chunk_char_len': 183,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
