from langchain_core.documents import Document

chunk = Document(
    page_content='법 제731조제1항에 따른 본인 확인 및 위조ㆍ변조 방지에 대한 신뢰성을 갖춘 전자문서는 다음 각 호의 요건을 모 두 갖춘 전자문서로 한다.',
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 68},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000102',
              'chunk_char_len': 78,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
