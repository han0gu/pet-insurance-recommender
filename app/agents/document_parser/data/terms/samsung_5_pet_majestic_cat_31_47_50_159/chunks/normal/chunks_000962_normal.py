from langchain_core.documents import Document

chunk = Document(
    page_content=('7) 한 발가락에 장해가 생기고 다른 발가락에 장해가 발생한 경우, 지급률은 각각 적용하여 합산한다. 8) 발가락 관절의 운동범위 측정은 '
 '장해평가시점의 「산업재해보상보험법 시행규 칙」 제47조 제1항 및 제3항의 정상인의 신체 각 관절에 대한 평균 운동가능 영역을 기준으로 '
 '정상각도 및 측정방법 등을 따른다.\n'
 '12. 흉 · 복부 장기 및 비뇨생식기의 장해\n'
 '가. 장해의 분류\n'
 '장 해 의 분 류 | 지급률(%)\n'
 '1) 심장 기능을 잃었을 때 | 100\n'
 '2) 흉복부장기 또는 비뇨생식기 기능을 잃었을 때 | 75'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 146},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage',
            'risk_domains': ['digestive', 'urinary']},
 'indexing': {'chunk_id': 'chunk_000962',
              'chunk_char_len': 282,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
