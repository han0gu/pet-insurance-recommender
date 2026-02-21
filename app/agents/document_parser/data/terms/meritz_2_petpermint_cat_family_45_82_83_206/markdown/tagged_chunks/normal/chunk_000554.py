from langchain_core.documents import Document

chunk = Document(
    page_content=('- 을 하나의 운동단위로 보며, 하나의 운동단위 내에\n'
 '- 서 여러 개의 척추체(척추뼈 몸통)에 압박골절이 발\n'
 '- 생한 경우에는 각 척추체(척추뼈 몸통)의 압박률을\n'
 '- 합산하고, 두 개 이상의 운동단위에서 장해가 발생\n'
 '- 한 경우에는 그 중 가장 높은 지급률을 적용한다.\n'
 '- 3) 척추(등뼈)의 장해는 퇴행성 기왕증 병변과 사고가\n'
 '- 그 증상을 악화시킨 부분만큼, 즉 이 사고와의 관여\n'
 '- 도를 산정하여 평가한다.\n'
 '- 4) 추간판탈출증으로 인한 신경 장해는 수술 또는 시술(비\n'
 '- 수술적 치료) 후 6개월 이상 지난 후에 평가한다.'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'head']},
 'indexing': {'chunk_id': 'chunk_000554',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
