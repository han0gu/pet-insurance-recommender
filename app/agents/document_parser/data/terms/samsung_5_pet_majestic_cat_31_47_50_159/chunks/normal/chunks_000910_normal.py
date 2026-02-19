from langchain_core.documents import Document

chunk = Document(
    page_content=("가) 척추체(척추뼈 몸통)의 만곡변화는 객관적인 측정방법(Cobb's Angle)에 따 라 골절이 발생한 척추체(척추뼈 몸통)의 상 · "
 '하 인접 정상 척추체(척추뼈 몸통)를 포함하여 측정하며, 생리적 정상만곡을 고려하여 평가한다. 나) 척추(등뼈)의 기형장해는 '
 '척추체(척추뼈 몸통)의 압박률, 골절의 부위 등을 기준으로 판정한다'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 140},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['joint', 'other']},
 'indexing': {'chunk_id': 'chunk_000910',
              'chunk_char_len': 183,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
