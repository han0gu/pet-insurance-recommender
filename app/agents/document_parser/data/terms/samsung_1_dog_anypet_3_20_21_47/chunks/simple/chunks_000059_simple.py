from langchain_core.documents import Document

chunk = Document(
    page_content=('제18조(계약의 무효)\n'
 '계약을 맺을 때에 보험사고가 이미 발생하였을 경우 이 계약은 무효로 합니다. 다만, 회사의 고의 또 는 과실로 계약이 무효로 된 경우와 '
 '회사가 승낙 전에 무효임을 알았거나 알 수 있었음에도 불구하고 보험료를 반환하지 않은 경우에는 보험료를 납입한 날의 다음날부터 '
 '반환일까지의 기간에 대하여 회 사는 보험개발원이 공시하는 보험계약대출이율을 연단위 복리로 계산한 금액을 더하여 돌려 드립니다.\n'
 '제19조(계약내용의 변경 등)'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 12},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000059',
              'chunk_char_len': 247,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
