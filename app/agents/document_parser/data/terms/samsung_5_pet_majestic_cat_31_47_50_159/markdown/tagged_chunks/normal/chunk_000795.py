from langchain_core.documents import Document

chunk = Document(
    page_content=("치료를 시행하였음에도 불구하고 골절부의 유합이 이루어지지 않는 '불유\n"
 '합\' 상태를 말하며, 골유합이 지연되는 지연유합은 제외한다.12) "가관절이 남아 약간의 장해를 남긴 때" 라 함은 요골과 척골 중 '
 '어느 한 뼈\n'
 '에 가관절이 남은 경우를 말한다. +\n'
 '13) "뼈에 기형을 남긴 때" 라 함은 상완골 또는 요골과 척골에 변형이 남아 정상\n'
 '에 비해 부정유합된 각 변형이 15° 이상인 경우를 말한다.- \n'
 '# 다. 지급률의 결정- \n'
 '- 1) 한 팔의 3대 관절 중 관절 하나에 기능장해가 생기고 다른 관절 하나에 기능장'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive', 'joint']},
 'indexing': {'chunk_id': 'chunk_000795',
              'chunk_char_len': 288,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
