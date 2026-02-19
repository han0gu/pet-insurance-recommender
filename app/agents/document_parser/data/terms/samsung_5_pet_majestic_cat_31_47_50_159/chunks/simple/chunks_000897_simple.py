from langchain_core.documents import Document

chunk = Document(
    page_content=('. 가) 언어평가상 자음정확도가 30% 미만인 경우 나) 전실어증, 운동성실어증(브로카실어증)으로 의사소통이 불가한 경우 8) "말하는 '
 '기능에 뚜렷한 장해를 남긴 때" 라 함은 아래의 경우 중 하나 이상에 해당되는 때를 말한다. 가) 언어평가상 자음정확도가 50% 미만인 '
 '경우 나) 언어평가상 표현언어지수 25 미만인 경우'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 140},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['head']},
 'indexing': {'chunk_id': 'chunk_000897',
              'chunk_char_len': 180,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
