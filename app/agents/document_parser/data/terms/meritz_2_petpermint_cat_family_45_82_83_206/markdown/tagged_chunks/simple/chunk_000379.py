from langchain_core.documents import Document

chunk = Document(
    page_content=('- ·보험금 지급금액\n'
 '- = [(410만원-3만원)×70%, 250만원] 중 적은금액\n'
 '- = 250만원(MRI,CT 및 내시경처치와 수술을 동시에\n'
 '- 하더라도 수술한도로 지급)\n'
 '\uf000 수술과 MRI,CT 및 내시경처치를 동일한 날에 시행한 경145우 수술한 날의 지급한도 내에서 보험금이 지급됩니다.\n'
 '\uf000 연간 1년 이내에 각각 다른 MRI,CT 및 내시경처치를 받\n'
 '은 경우 MRI,CT 및 내시경처치 의료행위 중 어느 하나의 의\n'
 '료행위가 연간 첫 번째로 발생한 때에는 제2항의 연간 첫'),
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
 'indexing': {'chunk_id': 'chunk_000379',
              'chunk_char_len': 269,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
