from langchain_core.documents import Document

chunk = Document(
    page_content=('13) "가관절이 남아 약간의 장해를 남긴 때" 라 함은 경골과 종아리뼈 중 어느 한 뼈에 가관절이 남은 경우를 말한다. 14) "뼈에 '
 '기형을 남긴 때" 라 함은 대퇴골 또는 경골에 기형이 남아 정상에 비해\n'
 '- 144 -\n'
 '부정유합된 각 변형이 15° 이상인 경우를 말한다.\n'
 '15) 다리 길이의 단축 또는 과신장은 스캐노그램(scanogram)을 통하여 측정한다.\n'
 '다. 지급률의 결정'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 145},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['joint', 'other']},
 'indexing': {'chunk_id': 'chunk_000947',
              'chunk_char_len': 213,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
