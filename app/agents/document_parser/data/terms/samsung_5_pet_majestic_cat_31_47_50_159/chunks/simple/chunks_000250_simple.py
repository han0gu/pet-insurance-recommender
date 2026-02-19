from langchain_core.documents import Document

chunk = Document(
    page_content=('<관련법규>\n'
 '[금융소비자보호에 관한 법률 제46조(청약의 철회)에서 정한 청약철회가능 기간] 일반금융소비자가 상법 제640조에 따른 보험증권을 받은 '
 '날부터 15일과 청약을 한 날부터 30일 중 먼저 도래하는 기간을 말합니다.\n'
 '③ 청약철회는 계약자가 전화로 신청하거나, 철회의사를 표시하기 위한 서면, 전자우편, 휴대전화 문자메시지 또는 이에 준하는 전자적 '
 '의사표시(이하 ‘서면 등’이라 합니다) 를 발송한 때 효력이 발생합니다. 계약자는 서면 등을 발송한 때에 그 발송 사실을 회'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 56},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000250',
              'chunk_char_len': 268,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
