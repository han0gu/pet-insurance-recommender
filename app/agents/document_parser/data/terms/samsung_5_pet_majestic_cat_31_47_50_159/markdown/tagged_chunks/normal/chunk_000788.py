from langchain_core.documents import Document

chunk = Document(
    page_content=('| 6) 한 팔의 3대 관절 중 관절 하나의 기능에 약간의 장해를 남긴 때 | 5 |\n'
 '| 7) 한 팔에 가관절이 남아 뚜렷한 장해를 남긴 때 | 20 |\n'
 '- 142 -8) 한 팔에 가관절이 남아 약간의 장해를 남긴 때\n'
 '9) 한 팔의 뼈에 기형을 남긴 때105# 나. 장해판정기준- 1) 골절부에 금속내고정물 등을 사용하였기 때문에 그것이 기능장해의 원인이 '
 '되\n'
 '- 는 때에는 그 내고정물 등이 제거된 후 장해를 평가한다. 단, 제거가 불가능한\n'
 '- 경우에는 고정물 등이 있는 상태에서 장해를 평가한다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'joint']},
 'indexing': {'chunk_id': 'chunk_000788',
              'chunk_char_len': 278,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
