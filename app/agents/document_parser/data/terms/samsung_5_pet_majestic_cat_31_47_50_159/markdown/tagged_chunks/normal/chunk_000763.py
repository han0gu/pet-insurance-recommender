from langchain_core.documents import Document

chunk = Document(
    page_content=('1) 외모에 뚜렷한 추상(추한 모습)을 남긴 때 15\n'
 '2) 외모에 약간의 추상(추한 모습)을 남긴 때 5# 나. 장해판정기준1) "외모" 란 얼굴(눈, 코, 귀, 입 포함), 머리, 목을 '
 '말한다.\n'
 '2) "추상(추한 모습)장해" 라 함은 성형수술(반흔성형술, 레이저치료 등 포함)을\n'
 '시행한 후에도 영구히 남게 되는 상태의 추상(추한 모습)을 말한다.- \n'
 '- - 139 -\n'
 '- 3) "추상(추한 모습)을 남긴 때" 라 함은 상처의 흔적, 화상 등으로 피부의 변색,'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other',
            'risk_domains': ['digestive', 'eye', 'head', 'skin']},
 'indexing': {'chunk_id': 'chunk_000763',
              'chunk_char_len': 257,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
