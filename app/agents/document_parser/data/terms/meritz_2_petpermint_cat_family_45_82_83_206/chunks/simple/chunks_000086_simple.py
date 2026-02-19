from langchain_core.documents import Document

chunk = Document(
    page_content=('며, 보험료 반환이 늦어진 기간에 대하여는 이 계약의 보험 계약대출이율을 연단위 복리로 계산한 금액을 더하여 지급 합니다. 다만, '
 '계약자가 제1회 보험료를 신용카드로 납입한 계약의 청약을 철회하는 경우에는 회사는 청약의 철회를 접 수한 날부터 3영업일 이내에 해당 '
 '신용카드회사로 하여금 대금청구를 하지 않도록 해야 하며, 이 경우 회사는 보험료 를 반환한 것으로 봅니다.'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 65},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000086',
              'chunk_char_len': 207,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
