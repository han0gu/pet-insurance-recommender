from langchain_core.documents import Document

chunk = Document(
    page_content=('이 유지되는 기간에는 언제든지 서면동의를 장래를 향하여\n'
 '철회할 수 있으며, 서면동의 철회로 계약이 해지되어 회사가79지급하여야 할 해약환급금이 있을 때에는 제35조(해약환급\n'
 '금) 제1항에 따른 해약환급금을 계약자에게 지급합니다.# 제32조의1(위법계약의 해지)\uf000 계약자는 ｢금융소비자보호에 관한 법률｣ '
 '제47조 및 관련\n'
 '규정이 정하는 바에 따라 계약체결에 대한 회사의 법위반사\n'
 '항이 있는 경우 계약체결일부터 5년 이내의 범위에서 계약\n'
 '자가 위반사항을 안 날로부터 1년 이내에 계약해지요구서에'),
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
 'indexing': {'chunk_id': 'chunk_000116',
              'chunk_char_len': 274,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
