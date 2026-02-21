from langchain_core.documents import Document

chunk = Document(
    page_content=('회복))\uf000 제29조(보험료의 납입이 연체되는 경우 납입최고(독촉)\n'
 '와 계약의 해지)에 따라 계약이 해지되었으나 해약환급금을\n'
 '받지 않은 경우(보험계약대출 등에 따라 해약환급금이 차감\n'
 '되었으나 받지 않은 경우 또는 해약환급금이 없는 경우를\n'
 '포함합니다) 계약자는 해지된 날부터 3년 이내에 회사가 정\n'
 '한 절차에 따라 계약의 부활(효력회복)을 청약할 수 있습니\n'
 '다. 회사가 부활(효력회복)을 승낙한 때에 계약자는 부활\n'
 '(효력회복)을 청약한 날까지의 연체된 보험료와 이에 대한\n'
 '연체된 이자(보장보험료에 대해서 평균공시이율+1%로 계산'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000108',
              'chunk_char_len': 293,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
