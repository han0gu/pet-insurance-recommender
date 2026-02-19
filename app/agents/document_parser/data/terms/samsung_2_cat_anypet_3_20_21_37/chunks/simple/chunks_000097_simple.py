from langchain_core.documents import Document

chunk = Document(
    page_content=('계약의 무효, 효력상실 또는 해지로 인하여 회사가 돌려드려야 할 보험료가 있을 때에는 계약자는 환급금을 청구하여야 하며, 회사는 청구일의 '
 "다음 날부터 지급일까지의 기간에 대하여 '보험개발원 이 공시하는 보험계약대출이율'을 연단위 복리로 계산한 금액을 더하여 지급합니다. ⑤ "
 "회사가 해지권을 행사하는 경우 제4항의 '청구일'은 회사의 해지 의사표시(서면, 전자우편, 휴대전 화 문자메시지 또는 이에 준하는 전자적 "
 '의사표시 포함)가 계약자 또는 그의 대리인에게 도달한 날 로 봅니다.\n'
 '제7관 분쟁의 조정 등\n'
 '제31조(분쟁의 조정)'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 35,
         'page': 18},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000097',
              'chunk_char_len': 294,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
