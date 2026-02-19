from langchain_core.documents import Document

chunk = Document(
    page_content=('제4조(입원의 정의와 장소)\n'
 '이 계약에 있어서 「입원」이라 함은 수의사가 상해 또는 질병의 치료가 필요하다고 인정한 경우로서, 자택 등에서의 치료가 곤란하여 '
 '동물병원에 입실하여 수의사의 관리 하에 치료에 전념하는 것을 말합니다.\n'
 '제5조(특별약관의 소멸)\n'
 '이 특별약관에서 정한 보상하는 손해가 더 이상 발생할 수 없는 경우에는 이 특별약관은 그 때부터 소멸되며, 이 경우 회사는「보험료 및 '
 '해약환급금 산출방법서」에서 정한 이 특별약관의 그 때까지 적립한 계약자적립액 및 미경과보험 료를 지급합니다.\n'
 '제6조(준용규정)'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 141},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000459',
              'chunk_char_len': 287,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
