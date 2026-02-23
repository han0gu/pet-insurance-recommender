from langchain_core.documents import Document

chunk = Document(
    page_content=('할하는 법원으로 합니다. 다만, 회사와 계약자가 합의하여\n'
 '관할법원을 달리 정할 수 있습니다.# 제41조(소멸시효)보험금청구권, 만기환급금청구권, 보험료반환청구권, 해약\n'
 '환급금청구권 및 계약자적립액 반환청구권은 3년간 행사하\n'
 '지 않으면 소멸시효가 완성됩니다.# 【소멸시효】소멸시효는 해당 청구권을 행사할 수 있는 때부터 진행합\n'
 '니다. 보험금 지급사유가 2023년 4월 1일에 발생하였음에\n'
 '도 2026년 4월 1일까지 보험금을 청구하지 않는 경우 소\n'
 '멸시효가 완성되어 보험금 등을 지급받지 못할 수 있습니'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'claim', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000130',
              'chunk_char_len': 280,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
