from langchain_core.documents import Document

chunk = Document(
    page_content=('- \uf000 보험수익자와 회사가 제1조(보험금의 지급사유)의 보험금 지급사유에 대해 합의\n'
 '- 하지 못할 때는 보험수익자와 회사가 함께 제3자를 정하고 그 제3자의 의견에\n'
 '- 따를 수 있습니다. 제3자는 의료법 제3조(의료기관)에 규정한 종합병원 소속 전\n'
 '- 문의 중에 정하며, 보험금 지급사유 판정에 드는 의료비용은 회사가 전액 부담\n'
 '# 합니다.# 제3조(수술의 정의와# 장소)- \uf000 이 특별약관에 있어서 "수술"이라 함은 병원 또는 의원의 의사, 치과의사 '
 '면허를'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000277',
              'chunk_char_len': 256,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
