from langchain_core.documents import Document

chunk = Document(
    page_content=('. 6) "발가락에 뚜렷한 장해를 남긴 때" 라 함은 첫째 발가락의 경우에 중족지관절 과 지관절의 굴신(굽히고 펴기)운동범위 합계가 정상 '
 '운동 가능영역의 1/2 이 하가 된 경우를 말하며, 다른 네 발가락에 있어서는 중족지관절의 신전운동범 위만을 평가하여 정상 운동범위의 '
 '1/2이하로 제한된 경우를 말한다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 146},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['joint']},
 'indexing': {'chunk_id': 'chunk_000961',
              'chunk_char_len': 171,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
