from langchain_core.documents import Document

chunk = Document(
    page_content=('사용처 등을 명시하고 설명합니다.\n'
 '\uf000 보험수익자와 회사가 보험금 지급사유에 대해 합의하지 못할 때는 보험수익자와 회사가 함께 제3자를 정하고 그 제 3자의 의견에 '
 '따를 수 있습니다. 제3자는 동물병원 소속의 수의사 중에서 정하며, 보험금 지급사유 판정에 드는 의료 비용은 회사가 전액 부담합니다.\n'
 '제6조(지급보험금의 계산)'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 90},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000208',
              'chunk_char_len': 180,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
