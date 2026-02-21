from langchain_core.documents import Document

chunk = Document(
    page_content=('- 주지 않았거나 계약자 또는 피보험자가 사실대로 알리\n'
 '- 는 것을 방해한 경우, 계약자 또는 피보험자에게 사실\n'
 '- 대로 알리지 않게 하였거나 부실한 사항을 알릴 것을\n'
 '- 권유했을 때. 다만, 보험설계사 등의 행위가 없었다\n'
 '- 하더라도 계약자 또는 피보험자가 사실대로 알리지 않\n'
 '- 거나 부실한 사항을 알렸다고 인정되는 경우에는 계약\n'
 '- 을 해지할 수 있습니다.\n'
 '\uf000 제1항에 따라 계약을 해지하였을 때에는 해약환급금을\n'
 '계약자에게 지급합니다.\n'
 '\uf000 제1항 제1호에 따른 계약의 해지가 손해발생 후에 이루'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000510',
              'chunk_char_len': 279,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
