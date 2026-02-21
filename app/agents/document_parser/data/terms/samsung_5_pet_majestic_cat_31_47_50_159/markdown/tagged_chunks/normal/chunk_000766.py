from langchain_core.documents import Document

chunk = Document(
    page_content=('- 나) 길이 5cm 이상의 추상 반흔(추한 모습의 흉터)\n'
 '- 다) 지름 2cm 이상의 조직함몰\n'
 '- 라) 코의 1/4 이상 결손\n'
 '2) 머리\n'
 '가) 손바닥 크기 1/2 이상의 반흔(흉터) 및 모발결손\n'
 '나) 머리뼈의 손바닥 크기 1/2 이상의 손상 및 결손\n'
 '3) 목\n'
 '가) 손바닥 크기 1/2 이상의 추상(추한 모습)- \n'
 '마. 손바닥 크기1) "손바닥 크기" 라 함은 해당 환자의 손가락을 제외한 손바닥의 크기를 말하며,12세 이상의 성인에서는 8× '
 '10cm(1/2 크기는 40cm2, 1/4 크기는 20cm2), 6~11'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['head']},
 'indexing': {'chunk_id': 'chunk_000766',
              'chunk_char_len': 288,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
