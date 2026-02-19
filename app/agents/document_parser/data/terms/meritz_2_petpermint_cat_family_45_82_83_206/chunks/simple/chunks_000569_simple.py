from langchain_core.documents import Document

chunk = Document(
    page_content=('제4조(입원의 정의와 장소)\n'
 '이 특별약관에 있어서 「입원」이라 함은 수의사가 상해 또 는 질병의 치료가 필요하다고 인정한 경우로서, 자택 등에 서의 치료가 곤란하여 '
 '동물병원에 입실하여 수의사의 관리 하에 치료에 전념하는 것을 말합니다.\n'
 '제5조(MRI,CT 및 내시경처치의 정의)'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 164},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000569',
              'chunk_char_len': 155,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
