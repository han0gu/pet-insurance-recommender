from langchain_core.documents import Document

chunk = Document(
    page_content=('- - 저축성 보험계약의 경우 적용이율 및 산출기준\n'
 '- - 유배당 보험계약의 경우 계약자 배당에 관한 사항\n'
 '- - 만기시 자동갱신되는 보험계약의 경우 자동갱신의 조건\n'
 '- - 그 밖에 약관에 기재된 보험계약의 중요사항\n'
 '# [통신판매계약]전화·우편·인터넷 등 통신수단을 이용하여 체결하는 계약을 말합니다.③ 회사가 제1항에 따라 제공될 약관 및 계약자 '
 '보관용 청약서를 청약할 때 계약자에게\n'
 '전달하지 않거나 약관의 중요한 내용을 설명하지 않은 때 또는 계약을 체결할 때 계'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000086',
              'chunk_char_len': 261,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
