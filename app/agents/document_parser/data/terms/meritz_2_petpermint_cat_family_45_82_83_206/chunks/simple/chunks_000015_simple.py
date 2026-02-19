from langchain_core.documents import Document

chunk = Document(
    page_content=('. \uf000【별표2(장해분류표)】에 해당되지 않는 후유장해는 피보 험자의 직업, 연령, 신분 또는 성별 등에 관계없이 신체의 '
 '장해정도에 따라【별표2(장해분류표)】의 구분에 준하여 지 급액을 결정합니다. \uf000 보험수익자와 회사가 제3조(보험금의 지급사유)의 '
 '보험 금 지급사유에 대해 합의하지 못할 때는 보험수익자와 회사 가 함께 제3자를 정하고 그 제3자의 의견에 따를 수 있습니 다. 제3자는 '
 '의료법 제3조(의료기관)에 규정한 종합병원 소 속 전문의 중에 정하며, 보험금 지급사유 판정에 드는 의료 비용은 회사가 전액 부담합니다'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 50},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000015',
              'chunk_char_len': 291,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
