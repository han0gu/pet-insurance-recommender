from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 반려동물의 나이 및 품종이 정정되기 이전에는 「나이 및 품종이 정정되기 전에 적용된 보험료율」의 「나이 및 품종이 정정된 후에 '
 '적용해 야할 보험료율」에 대한 비율에 따라 보험금을 삭감하여 지 급합니다.'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 97},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000245',
              'chunk_char_len': 118,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.9}},
)
