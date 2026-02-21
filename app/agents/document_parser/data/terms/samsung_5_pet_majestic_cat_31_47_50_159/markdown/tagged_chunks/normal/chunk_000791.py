from langchain_core.documents import Document

chunk = Document(
    page_content=('- 능영역을 기준으로 정상각도 및 측정방법 등을 따른다.\n'
 '- 나) 관절기능장해를 표시할 경우 장해부위의 장해각도와 정상부위의 측정치를\n'
 '- 동시에 판단하여 장해상태를 명확히 한다. 단, 관절기능장해가 신경손상으\n'
 '- 로 인한 경우에는 운동범위 측정이 아닌 근력 및 근전도 검사를 기준으로\n'
 '- 평가한다.\n'
 '7) "관절 하나의 기능을 완전히 잃었을 때" 라 함은 아래의 경우 중 하나에 해당\n'
 '하는 경우를 말한다.- 가) 완전 강직(관절굳음)\n'
 '- 나) 근전도 검사상 완전손상(complete injury) 소견이 있으면서 도수근력검사'),
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
            'risk_domains': ['digestive', 'head', 'joint']},
 'indexing': {'chunk_id': 'chunk_000791',
              'chunk_char_len': 292,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
