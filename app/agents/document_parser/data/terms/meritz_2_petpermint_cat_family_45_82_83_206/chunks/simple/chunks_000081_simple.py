from langchain_core.documents import Document

chunk = Document(
    page_content=('. \uf000 제5항의「청약일로부터 5년이 지나는 동안」이라 함은 제29조(보험료의 납입이 연체되는 경우 납입최고(독촉)와 계약의 '
 '해지)에서 정한 계약의 해지가 발생하지 않은 경우 를 말합니다. \uf000 제30조(보험료의 납입을 연체하여 해지된 계약의 부활 '
 '(효력회복))에서 정한 계약의 부활이 이루어진 경우 부활을 청약한 날을 제5항의 청약일로 하여 적용합니다.'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 64},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000081',
              'chunk_char_len': 196,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
