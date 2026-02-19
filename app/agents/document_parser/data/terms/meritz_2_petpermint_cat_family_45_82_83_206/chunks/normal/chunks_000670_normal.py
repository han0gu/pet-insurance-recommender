from langchain_core.documents import Document

chunk = Document(
    page_content=('3) 척추(등뼈)의 장해는 퇴행성 기왕증 병변과 사고가 그 증상을 악화시킨 부분만큼, 즉 이 사고와의 관여 도를 산정하여 평가한다. 4) '
 '추간판탈출증으로 인한 신경 장해는 수술 또는 시술(비 수술적 치료) 후 6개월 이상 지난 후에 평가한다. 5) 신경학적 검사상 나타난 '
 '저린감이나 방사통 등 신경자극 증상의 원인으로 CT, MRI 등 영상검사에서 추간판탈출증 이 확인된 경우를 추간판탈출증으로 진단하며, '
 '수술 여 부에 관계없이 운동장해 및 기형장해로 평가하지 않는 다'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 186},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage',
            'risk_domains': ['joint', 'head', 'other']},
 'indexing': {'chunk_id': 'chunk_000670',
              'chunk_char_len': 262,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.85}},
)
