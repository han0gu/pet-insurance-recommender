from langchain_core.documents import Document

chunk = Document(
    page_content=('다. 뚜렷한 추상(추한 모습)\n'
 '1) 얼굴\n'
 '가) 손바닥 크기 1/2 이상의 추상(추한 모습) 나) 길이 10cm 이상의 추상 반흔(추한 모습의 흉터) 다) 지름 5cm 이상의 '
 '조직함몰 라) 코의 1/2 이상 결손\n'
 '2) 머리 가) 손바닥 크기 이상의 반흔(흉터) 및 모발결손 나) 머리뼈의 손바닥 크기 이상의 손상 및 결손 3) 목 가) 손바닥 크기 '
 '이상의 추상(추한 모습)\n'
 '라. 약간의 추상(추한 모습)\n'
 '1) 얼굴'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 140},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion',
            'risk_domains': ['skin', 'head', 'eye', 'other']},
 'indexing': {'chunk_id': 'chunk_000905',
              'chunk_char_len': 230,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
