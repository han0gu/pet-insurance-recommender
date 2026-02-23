from langchain_core.documents import Document

chunk = Document(
    page_content=('제 6 관 계약의 해지 및 해약환급금등# 제31조(계약자의 임의해지 및 피보험자의 서면동의철회)\uf000 계약자는 계약이 소멸하기 '
 '전에는 언제든지 계약을 해지할 수 있으며, 이 경우 회\n'
 '사는 제34조(해약환급금) 제1항에 따른 해약환급금을 계약자에게 지급합니다.\n'
 '\uf000 제21조(계약의 무효)에 따라 사망을 보험금 지급사유로 하는 계약에서 서면으로 동\n'
 '의를 한 피보험자는 계약의 효력이 유지되는 기간에는 언제든지 서면동의를 장래를\n'
 '향하여 철회할 수 있으며, 서면동의 철회로 계약이 해지되어 회사가 지급하여야 할'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000176',
              'chunk_char_len': 279,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
