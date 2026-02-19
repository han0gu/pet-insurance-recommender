from langchain_core.documents import Document

chunk = Document(
    page_content=('12) "가관절이 남아 약간의 장해를 남긴 때" 라 함은 요골과 척골 중 어느 한 뼈 에 가관절이 남은 경우를 말한다. + 13) "뼈에 '
 '기형을 남긴 때" 라 함은 상완골 또는 요골과 척골에 변형이 남아 정상 에 비해 부정유합된 각 변형이 15° 이상인 경우를 말한다.\n'
 '다. 지급률의 결정\n'
 '1) 한 팔의 3대 관절 중 관절 하나에 기능장해가 생기고 다른 관절 하나에 기능장 해가 발생한 경우 지급률은 각각 적용하여 합산한다. '
 '2) 1상지(팔과 손가락)의 후유장해지급률은 원칙적으로 각각 합산하되, 지급률은 60% 한도로 한다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 143},
 'term_type': 'special',
 'clause': {'clause_type': 'limit', 'risk_domains': ['digestive', 'joint']},
 'indexing': {'chunk_id': 'chunk_000935',
              'chunk_char_len': 294,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
