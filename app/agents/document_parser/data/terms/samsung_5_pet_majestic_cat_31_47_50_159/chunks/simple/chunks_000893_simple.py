from langchain_core.documents import Document

chunk = Document(
    page_content=('에 해당되는 때를 말한다.\n'
 '가) 뚜렷한 개구운동 제한 또는 뚜렷한 저작운동 제한으로 미음 또는 이에 준하 는 정도의 음식물(죽 등) 이외는 섭취하지 못하는 경우 '
 '나) 위·아래턱(상·하악)의 가운데 앞니(중절치)간 최대 개구운동이 1cm 이하로 제한되는 경우 다) 위·아래턱(상·하악)의 '
 '부정교합(전방, 측방)이 1.5cm 이상인 경우 라) 1개 이하의 치아만 교합되는 상태 마) 연하기능검사(비디오 투시검사)상 연하장애가 '
 '있고, 유동식 섭취 시 흡인이 발생하고 연식 외에는 섭취가 불가능한 상태'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 140},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion',
            'risk_domains': ['other', 'dental', 'digestive']},
 'indexing': {'chunk_id': 'chunk_000893',
              'chunk_char_len': 276,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
