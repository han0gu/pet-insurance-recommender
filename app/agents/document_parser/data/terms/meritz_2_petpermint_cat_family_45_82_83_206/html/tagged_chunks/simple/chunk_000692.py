from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:14px'>145</footer><p id='94' data-category='paragraph' "
 "style='font-size:18px'>우 수술한 날의 지급한도 내에서 보험금이 지급됩니다.<br>\uf000 연간 1년 이내에 각각 "
 '다른 MRI,CT 및 내시경처치를 받<br>은 경우 MRI,CT 및 내시경처치 의료행위 중 어느 하나의 의<br>료행위가 연간 첫 번째로 '
 '발생한 때에는 제2항의 연간 첫<br>번째 지급한도 내에서 보험금을 지급하며 연간 첫 번째 의<br>료행위 이후에 발생한 MRI,CT 및'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000692',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
