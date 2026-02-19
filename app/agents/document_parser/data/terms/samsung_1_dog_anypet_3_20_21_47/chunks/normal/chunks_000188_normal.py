from langchain_core.documents import Document

chunk = Document(
    page_content=('제4조(보험기간의 설정)\n'
 '회사는 단체계약 특별약관 제4조(보험의 목적의 증가 감소 또는 교체) 제2항에도 불구하고 새로이 증 가 또는 교체되는 피보험자의 '
 '보험기간은 계약자가 요청하는 기간으로 할 수 있습니다. 다만, 이 계약 기간 중 피보험자 감소의 경우 피보험자가 소속단체를 '
 '탈퇴(퇴사)하는 즉시 당해 피보험자의 계약은 해지된 것으로 합니다.\n'
 '제5조(준용규정)\n'
 '이 추가특별약관에서 정하지 않은 사항은 보통약관 및 해당특별약관을 따릅니다.'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 37},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000188',
              'chunk_char_len': 245,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
