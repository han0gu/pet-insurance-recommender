from langchain_core.documents import Document

chunk = Document(
    page_content=('다) 지름 5cm 이상의 조직함몰\n'
 '라) 코의 1/2이상 결손# 2) 머리가) 손바닥 크기 이상의 반흔(흉터) 및 모발결손184# 나) 머리뼈의 손바닥 크기 이상의 손상 및 '
 '결손# 3) 목# 손바닥 크기 이상의 추상(추한 모습)# 라. 약간의 추상(추한 모습)# 1) 얼굴가) 손바닥 크기 1/4 이상의 '
 '추상(추한 모습)\n'
 '나) 길이 5cm 이상의 추상반흔(추한 모습의 흉터)\n'
 '다) 지름 2cm 이상의 조직함몰\n'
 '라) 코의 1/4이상 결손# 2) 머리가) 손바닥 크기 1/2 이상의 반흔(흉터) 및 모발결손'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['head']},
 'indexing': {'chunk_id': 'chunk_000548',
              'chunk_char_len': 280,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
