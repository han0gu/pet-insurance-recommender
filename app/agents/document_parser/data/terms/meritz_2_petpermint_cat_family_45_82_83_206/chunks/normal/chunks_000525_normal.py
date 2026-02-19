from langchain_core.documents import Document

chunk = Document(
    page_content=('\uf000 수술과 MRI,CT 및 내시경처치를 동일한 날에 시행한 경 우 수술한 날의 지급한도 내에서 보험금이 지급됩니다. \uf000 '
 '연간 1년 이내에 각각 다른 MRI,CT 및 내시경처치를 받 은 경우 MRI,CT 및 내시경처치 의료행위 중 어느 하나의 의 료행위가 '
 '연간 첫 번째로 발생한 때에는 제2항의 연간 첫 번째 지급한도 내에서 보험금을 지급하며 연간 첫 번째 의 료행위 이후에 발생한 '
 'MRI,CT 및 내시경처치에 대하여 제2 항의 연간 두번째 이상 지급한도 내에서 보험금을 지급합니 다'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 155},
 'term_type': 'special',
 'clause': {'clause_type': 'limit', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000525',
              'chunk_char_len': 269,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
