from langchain_core.documents import Document

chunk = Document(
    page_content=('제36조(보험계약대출)\n'
 '\uf000 계약자는 이 계약의 해약환급금 범위 내에서 회사가 정 한 방법에 따라 대출(이하「보험계약대출」이라 합니다)을 받을 수 '
 '있습니다. 그러나, 순수보장성보험 등 보험상품의 종류에 따라 보험계약대출이 제한될 수도 있습니다. \uf000 계약자는 제1항에 따른 '
 '보험계약대출금과 그 이자를 언 제든지 상환할 수 있으며 상환하지 않은 때에는 회사는 보 험금, 해약환급금 등의 지급사유가 발생한 날에 '
 '지급금에서'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 77},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000154',
              'chunk_char_len': 231,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
