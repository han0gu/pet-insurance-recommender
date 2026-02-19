from langchain_core.documents import Document

chunk = Document(
    page_content=('제33조(소멸시효)\n'
 '보험금청구권, 보험료 또는 환급금 반환청구권은 3년간 행사하지 않으면 소멸시효가 완성됩니다.\n'
 '【소멸시효】 일정기간 행사하지 않으면 권리를 소멸시키는 제도입니다. 소멸시효는 권리를 행사할 수 있는 때로부터 진행합니다.\n'
 '【예시】 보험금 지급사유가 2022년 1월 1일에 발생하였음에도 2025년 1월 1일까지 보험금을 청구하지 않는 경우 소멸시효가 완성되어 '
 '보험금을 지급받지 못 할 수 있습니다. 다만, 2025년 1월 1일이 토요일 또는 공휴일 일 경우 그 다음 첫 영업일에 소멸시효가 '
 '완성됩니다.'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 19},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000104',
              'chunk_char_len': 288,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
