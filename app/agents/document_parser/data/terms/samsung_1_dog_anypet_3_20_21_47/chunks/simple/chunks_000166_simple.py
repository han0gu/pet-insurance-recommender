from langchain_core.documents import Document

chunk = Document(
    page_content=('③ 보험기간동안 이 보험의 보험요율이 변경된 경우라도 이 특별약관에 따라 납입하는 분납보험료는 변경 적용하지 않습니다. 다만, 보통약관 '
 '제13조(계약 후 알릴 의무)에 따라 보험료가 변경된 경우 에는 예외로 합니다.\n'
 '제2조(준용규정)\n'
 '이 특별약관에 정하지 않은 사항은 보통약관을 따릅니다.'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 32},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000166',
              'chunk_char_len': 161,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
