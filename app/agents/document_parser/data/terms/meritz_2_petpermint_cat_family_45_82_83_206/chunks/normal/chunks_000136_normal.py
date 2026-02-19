from langchain_core.documents import Document

chunk = Document(
    page_content=('\uf000 제29조(보험료의 납입이 연체되는 경우 납입최고(독촉) 와 계약의 해지)에 따라 계약이 해지되었으나 해약환급금을 받지 않은 '
 '경우(보험계약대출 등에 따라 해약환급금이 차감 되었으나 받지 않은 경우 또는 해약환급금이 없는 경우를 포함합니다) 계약자는 해지된 날부터 '
 '3년 이내에 회사가 정 한 절차에 따라 계약의 부활(효력회복)을 청약할 수 있습니 다'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 74},
 'term_type': 'basic',
 'clause': {'clause_type': 'renewal', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000136',
              'chunk_char_len': 195,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
