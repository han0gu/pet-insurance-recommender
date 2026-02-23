from langchain_core.documents import Document

chunk = Document(
    page_content=('결정합니다.<br>\uf000【별표2(장해분류표)】에 해당되지 않는 후유장해는 피보<br>험자의 직업, 연령, 신분 또는 성별 등에 '
 '관계없이 신체의<br>장해정도에 따라【별표2(장해분류표)】의 구분에 준하여 지<br>급액을 결정합니다.<br>\uf000 보험수익자와 '
 '회사가 제3조(보험금의 지급사유)의 보험<br>금 지급사유에 대해 합의하지 못할 때는 보험수익자와 회사<br>가 함께 제3자를 정하고 그 '
 '제3자의 의견에 따를 수 있습니<br>다'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000024',
              'chunk_char_len': 238,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
