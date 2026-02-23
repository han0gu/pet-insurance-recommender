from langchain_core.documents import Document

chunk = Document(
    page_content=('는 질병의 치료가 필요하다고 인정한 경우로서, 자택 등에\n'
 '서의 치료가 곤란하여 동물병원에 입실하여 수의사의 관리\n'
 '하에 치료에 전념하는 것을 말합니다.# 제5조(MRI,CT 및 내시경처치의 정의)\uf000 이 특별약관에 있어서 MRI,CT 및 '
 '내시경처치라 함은 자\n'
 '기공명영상(MRI), 전산화단층촬영(CT) 및 내시경처치를 말\n'
 '합니다.\n'
 '\uf000 제1항의 자기공명영상(MRI)이라 함은 제1조(보험금의 지\n'
 '급사유)에서 정한 수의사에 의하여 진단 및 치료가 필요하\n'
 '다고 인정된 경우로서 수의사의 관리 하에 자기공명영상'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'definition', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000454',
              'chunk_char_len': 279,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
