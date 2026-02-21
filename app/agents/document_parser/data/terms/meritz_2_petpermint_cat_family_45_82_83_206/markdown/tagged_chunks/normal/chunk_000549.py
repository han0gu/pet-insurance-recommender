from langchain_core.documents import Document

chunk = Document(
    page_content=('나) 머리뼈의 손바닥 크기 1/2 이상의 손상 및 결손# 3) 목손바닥 크기 1/2 이상의 추상(추한 모습)# 마. 손바닥 크기“손바닥 '
 '크기”라 함은 해당 환자의 손가락을 제외한 손\n'
 '바닥의 크기를 말하며, 12세 이상의 성인에서는 8×10㎝\n'
 '(1/2 크기는 40㎠, 1/4 크기는 20㎠), 6∼11세의 경우는\n'
 '6×8㎝(1/2 크기는 24㎠, 1/4 크기는 12㎠), 6세 미만의\n'
 '경우는 4×6㎝(1/2 크기는 12㎠, 1/4 크기는 6㎠)로 간\n'
 '주한다.# 6. 척추(등뼈)의 장해# 가. 장해의 분류| 장해의 분류 | 지급률 |'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive', 'head']},
 'indexing': {'chunk_id': 'chunk_000549',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
