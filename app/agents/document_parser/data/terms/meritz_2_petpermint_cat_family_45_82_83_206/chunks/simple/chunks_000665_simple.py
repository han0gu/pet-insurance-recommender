from langchain_core.documents import Document

chunk = Document(
    page_content='9) 추간판탈출증으로 인한 약간의 신경 장해 | 10\n나. 장해판정기준\n1) 척추(등뼈)는 경추에서 흉추, 요추, 제1천추까지를 동일',
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 185},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['joint', 'other']},
 'indexing': {'chunk_id': 'chunk_000665',
              'chunk_char_len': 74,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.85}},
)
