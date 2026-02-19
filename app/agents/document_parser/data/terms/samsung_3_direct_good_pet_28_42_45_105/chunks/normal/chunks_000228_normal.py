from langchain_core.documents import Document

chunk = Document(
    page_content=('- 청약의 철회에 관한 사항\n'
 '- 계약의 해지 및 해제\n'
 '- 분쟁조정 절차에 관한 사항 - 예금자보호법에 따른 보호여부 - 환급금에 관한 사항 - 고지의무 및 통지의무 위반의 효과 - 저축성 '
 '보험계약의 경우 적용이율 및 산출기준 - 유배당 보험계약의 경우 계약자 배당에 관한 사항 - 만기시 자동갱신되는 보험계약의 경우 '
 '자동갱신의 조건 - 그 밖에 약관에 기재된 보험계약의 중요사항\n'
 '[통신판매계약]\n'
 '전화·우편·인터넷 등 통신수단을 이용하여 체결하는 계약을 말합니다.'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 67,
         'page': 52},
 'term_type': 'special',
 'clause': {'clause_type': 'renewal', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000228',
              'chunk_char_len': 258,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
