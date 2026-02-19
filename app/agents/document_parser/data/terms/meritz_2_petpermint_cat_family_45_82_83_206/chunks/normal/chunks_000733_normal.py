from langchain_core.documents import Document

chunk = Document(
    page_content=('10) 약간의 치매 : CDR 척도 2점 | 40\n'
 '11) 심한 뇌전증 발작이 남았을 때 | 70\n'
 '12) 뚜렷한 뇌전증 발작이 남았을 때 | 40\n'
 '13) 약간의 뇌전증 발작이 남았을 때 | 10\n'
 '나. 장해판정기준\n'
 '1) 신경계'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 201},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['head', 'other']},
 'indexing': {'chunk_id': 'chunk_000733',
              'chunk_char_len': 124,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.9}},
)
