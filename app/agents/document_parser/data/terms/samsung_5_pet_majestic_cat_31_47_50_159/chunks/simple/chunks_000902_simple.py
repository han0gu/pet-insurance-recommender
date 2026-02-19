from langchain_core.documents import Document

chunk = Document(
    page_content=('5. 외모의 추상(추한 모습)장해\n'
 '가. 장해의 분류\n'
 '장 해 의 분 류 지급률(%) 1) 외모에 뚜렷한 추상(추한 모습)을 남긴 때 15 2) 외모에 약간의 추상(추한 모습)을 남긴 때 5\n'
 '나. 장해판정기준\n'
 '1) "외모" 란 얼굴(눈, 코, 귀, 입 포함), 머리, 목을 말한다. 2) "추상(추한 모습)장해" 라 함은 성형수술(반흔성형술, '
 '레이저치료 등 포함)을 시행한 후에도 영구히 남게 되는 상태의 추상(추한 모습)을 말한다.\n'
 '- 139 -'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 140},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage',
            'risk_domains': ['skin',
                             'head',
                             'eye',
                             'dental',
                             'joint',
                             'digestive',
                             'urinary',
                             'other']},
 'indexing': {'chunk_id': 'chunk_000902',
              'chunk_char_len': 248,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
