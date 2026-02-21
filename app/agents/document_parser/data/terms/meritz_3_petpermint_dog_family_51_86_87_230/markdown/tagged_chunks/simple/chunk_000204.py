from langchain_core.documents import Document

chunk = Document(
    page_content=('항의 반려동물보험 상품으로 재가입하는 것으로 하며, 기존\n'
 '계약은 해지됩니다. 다만, 계약자가 재가입을 원하지 않는\n'
 '경우에는 해당 시점으로부터 계약은 해지됩니다(단, 최초연\n'
 '장된 날로부터 90일 이전에는 계약을 취소 또는 해지할 수\n'
 '있습니다.)\uf000 제8항 내지 제10항에 따라 계약이 해지된 경우 회사는\n'
 '\uf000\n'
 '보통약관 제35조(해약환급금) 제1항에 따른 해약환급금을\n'
 '계약자에게 지급합니다.제16조(제1회 보험료 및 회사의 보장개시)\uf000 회사는 계약의 청약을 승낙하고 제1회 보험료를 받은 때'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000204',
              'chunk_char_len': 270,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
