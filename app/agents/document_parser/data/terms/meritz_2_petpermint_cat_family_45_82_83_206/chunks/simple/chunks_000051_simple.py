from langchain_core.documents import Document

chunk = Document(
    page_content=('【 계약 전 알릴 의무 】\n'
 '상법 제651조(고지의무위반으로 인한 계약해지)에서 정 하고 있는 의무. 계약자나 피보험자는 청약할 때에 회사 가 청약서에서 질문한 '
 '중요한 사항에 대해 사실대로 알 려야 하며, 위반하는 경우 계약의 해지 또는 보험금 부 지급 등 불이익을 당할 수 있습니다.\n'
 '【상법 제651조(고지의무위반으로 인한 계약해지)】'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 58},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000051',
              'chunk_char_len': 187,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
