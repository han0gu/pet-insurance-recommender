from langchain_core.documents import Document

chunk = Document(
    page_content=('이하의 천골 및 미골은 체간골의 장해로 평가한다.\n'
 '2) 척추(등뼈)의 기형장해는 척추체(척추뼈 몸통을 말하며, 횡돌기 및 극돌기는 제\n'
 '외한다. 이하 이 신체부위에서 같다)의 압박률 또는 척추체(척추뼈 몸통)의 만\n'
 '곡 정도에 따라 평가한다.- \n'
 "가) 척추체(척추뼈 몸통)의 만곡변화는 객관적인 측정방법(Cobb's Angle)에 따\n"
 '라 골절이 발생한 척추체(척추뼈 몸통)의 상 · 하 인접 정상 척추체(척추뼈\n'
 '몸통)를 포함하여 측정하며, 생리적 정상만곡을 고려하여 평가한다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000769',
              'chunk_char_len': 264,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
