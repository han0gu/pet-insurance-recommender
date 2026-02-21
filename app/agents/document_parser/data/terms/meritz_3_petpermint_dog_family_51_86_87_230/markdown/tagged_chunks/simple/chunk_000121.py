from langchain_core.documents import Document

chunk = Document(
    page_content=('니다.\uf000 회사가 제1항에 따라 계약을 해지한 경우 회사는 그 취\n'
 '지를 계약자에게 통지하고 제35조(해약환급금) 제1항에 따\n'
 '른 해약환급금을 지급합니다.# 제34조(회사의 파산선고와 해지)\uf000 회사가 파산의 선고를 받은 때에는 계약자는 계약을 해\n'
 '지할 수 있습니다.\n'
 '\uf000 제1항의 규정에 따라 해지하지 않은 계약은 파산선고 후\n'
 '3개월이 지난 때에는 그 효력을 잃습니다.\n'
 '\uf000 제1항의 규정에 따라 계약이 해지되거나 제2항의 규정에\n'
 '따라 계약이 효력을 잃는 경우에 회사는 제35조(해약환급'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000121',
              'chunk_char_len': 268,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
