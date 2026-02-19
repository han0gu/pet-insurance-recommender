from langchain_core.documents import Document

chunk = Document(
    page_content=('제40조(관할법원)\n'
 '이 계약에 관한 소송 및 민사조정은 계약자의 주소지를 관 할하는 법원으로 합니다. 다만, 회사와 계약자가 합의하여 관할법원을 달리 정할 '
 '수 있습니다.\n'
 '제41조(소멸시효)\n'
 '보험금청구권, 만기환급금청구권, 보험료반환청구권, 해약 환급금청구권 및 계약자적립액 반환청구권은 3년간 행사하 지 않으면 소멸시효가 '
 '완성됩니다.\n'
 '【소멸시효】'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 79},
 'term_type': 'basic',
 'clause': {'clause_type': 'claim', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000163',
              'chunk_char_len': 193,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
