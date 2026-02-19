from langchain_core.documents import Document

chunk = Document(
    page_content=('제7조(보험기간의 설정)\n'
 '회사는 새로이 증가 또는 교체되는 피보험자의 보험기간은 계약자가 요청하는 기간으로 합니다. 다만, 이 계약기간 중 피보험자 감소의 경우 '
 '당해 피보험자의 계약은 해지된 것으로 합니다.\n'
 '제8조(적용특칙)\n'
 '회사는 계약자에게만 보험증권을 드립니다.\n'
 '제9조(준용규정)\n'
 '이 추가특별약관에 정하지 않은 사항은 보통약관 및 해당특별약관을 따릅니다.\n'
 '[양식1]\n'
 '피보험자명 | 주민등록번호 (필요시) | 주소 (필요시) | 전화번호 (필요시) | 상품구입일 | 날인'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 42},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000207',
              'chunk_char_len': 264,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
