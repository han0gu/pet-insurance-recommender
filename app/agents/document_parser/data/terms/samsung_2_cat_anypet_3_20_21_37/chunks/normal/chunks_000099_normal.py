from langchain_core.documents import Document

chunk = Document(
    page_content=('제32조(관할법원)\n'
 '이 계약에 관한 소송 및 민사조정은 계약자의 주소지를 관할하는 법원으로 합니다. 다만, 회사와 계약 자가 합의하여 관할법원을 달리 정할 '
 '수 있습니다.\n'
 '제33조(소멸시효)\n'
 '보험금청구권, 보험료 또는 환급금 반환청구권은 3년간 행사하지 않으면 소멸시효가 완성됩니다.\n'
 '【소멸시효】 일정기간 행사하지 않으면 권리를 소멸시키는 제도입니다. 소멸시효는 권리를 행사할 수 있는 때로부터 진행합니다.'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 35,
         'page': 18},
 'term_type': 'basic',
 'clause': {'clause_type': 'claim', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000099',
              'chunk_char_len': 226,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
