from langchain_core.documents import Document

chunk = Document(
    page_content=('\uf000 제3항에 따라 계약이 취소된 경우에는 회사는 이미 납입 한 보험료를 계약자에게 돌려 드리며, 보험료를 받은 기간 에 대하여 '
 '보험계약대출이율을 연단위 복리로 계산한 금액 을 더하여 지급합니다.\n'
 '【보험계약대출이율】\n'
 '계약자는 해당 계약의 해약환급금 범위내에서 회사가 정 한 방법에 따라 대출을 받을 수 있는데, 이를「보험계약 대출」이라 합니다. 이 때 '
 '적용되는 이율을「보험계약대'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 66},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000095',
              'chunk_char_len': 210,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
