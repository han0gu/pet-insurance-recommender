from langchain_core.documents import Document

chunk = Document(
    page_content=('- 6) "발가락에 뚜렷한 장해를 남긴 때" 라 함은 첫째 발가락의 경우에 중족지관절\n'
 '- 과 지관절의 굴신(굽히고 펴기)운동범위 합계가 정상 운동 가능영역의 1/2 이\n'
 '- 하가 된 경우를 말하며, 다른 네 발가락에 있어서는 중족지관절의 신전운동범\n'
 '- 위만을 평가하여 정상 운동범위의 1/2이하로 제한된 경우를 말한다.\n'
 '- 7) 한 발가락에 장해가 생기고 다른 발가락에 장해가 발생한 경우, 지급률은 각각\n'
 '- 적용하여 합산한다.\n'
 '- 8) 발가락 관절의 운동범위 측정은 장해평가시점의 「산업재해보상보험법 시행규'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive', 'joint']},
 'indexing': {'chunk_id': 'chunk_000821',
              'chunk_char_len': 283,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
