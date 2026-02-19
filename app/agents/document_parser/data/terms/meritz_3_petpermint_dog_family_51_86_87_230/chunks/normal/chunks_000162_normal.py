from langchain_core.documents import Document

chunk = Document(
    page_content=('제41조(소멸시효)\n'
 '보험금청구권, 만기환급금청구권, 보험료반환청구권, 해약 환급금청구권 및 계약자적립액 반환청구권은 3년간 행사하 지 않으면 소멸시효가 '
 '완성됩니다.\n'
 '【소멸시효】\n'
 '소멸시효는 해당 청구권을 행사할 수 있는 때부터 진행합 니다. 보험금 지급사유가 2023년 4월 1일에 발생하였음에 도 2026년 4월 '
 '1일까지 보험금을 청구하지 않는 경우 소 멸시효가 완성되어 보험금 등을 지급받지 못할 수 있습니 다.\n'
 '제42조(약관의 해석)'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 83},
 'term_type': 'basic',
 'clause': {'clause_type': 'claim', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000162',
              'chunk_char_len': 244,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
