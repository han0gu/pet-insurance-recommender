from langchain_core.documents import Document

chunk = Document(
    page_content=('항목 | 자기부담금 | 지급 한도\n'
 '입원 의료비 | 입원 중 수술을 하지 않은 날의 경우 | 1일당 3만원/5만원 중 보험증권에 기재된 자기부담금 | 1일당 30만원\n'
 '입원 중 수술을 한 날의 경우 | 수술당일에 한하여 1일당 200만원'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 114},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000331',
              'chunk_char_len': 130,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
