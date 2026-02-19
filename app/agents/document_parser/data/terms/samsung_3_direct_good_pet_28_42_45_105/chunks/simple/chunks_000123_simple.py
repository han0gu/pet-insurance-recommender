from langchain_core.documents import Document

chunk = Document(
    page_content=('제6관 계약의 해지 및 해약환급금 등\n'
 '제30조 (계약자의 임의해지 및 피보험자의 서면동의 철회권)\n'
 '① 계약자는 계약이 소멸하기 전에는 언제든지 계약을 해지할 수 있으며, 이 경우 회사는 제33조(해약환급금) 제1항에 따른 해약환급금을 '
 '계약자에게 지급합니다. ② 제20조(계약의 무효)에 따라 사망을 보험금 지급사유로 하는 계약에서 서면으로 동의 를 한 피보험자는 계약의 '
 '효력이 유지되는 기간에는 언제든지 서면동의를 장래를 향'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 39},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000123',
              'chunk_char_len': 237,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
