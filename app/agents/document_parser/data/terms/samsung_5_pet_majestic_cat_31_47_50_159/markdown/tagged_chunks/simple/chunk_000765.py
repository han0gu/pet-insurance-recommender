from langchain_core.documents import Document

chunk = Document(
    page_content=('- 산정한다.\n'
 '# 다. 뚜렷한 추상(추한 모습)# 1) 얼굴가) 손바닥 크기 1/2 이상의 추상(추한 모습)\n'
 '나) 길이 10cm 이상의 추상 반흔(추한 모습의 흉터)\n'
 '다) 지름 5cm 이상의 조직함몰\n'
 '라) 코의 1/2 이상 결손- \n'
 '- 2) 머리\n'
 '- 가) 손바닥 크기 이상의 반흔(흉터) 및 모발결손\n'
 '- 나) 머리뼈의 손바닥 크기 이상의 손상 및 결손\n'
 '- 3) 목\n'
 '- 가) 손바닥 크기 이상의 추상(추한 모습)\n'
 '# 라. 약간의 추상(추한 모습)1) 얼굴- 가) 손바닥 크기 1/4 이상의 추상(추한 모습)'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['head']},
 'indexing': {'chunk_id': 'chunk_000765',
              'chunk_char_len': 282,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
