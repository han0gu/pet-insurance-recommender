from langchain_core.documents import Document

chunk = Document(
    page_content=('② 지급사유 관련 용어\n'
 '1. 상해: 보험기간 중에 발생한 급격하고도 우연한 외래의 사고로 반려묘에 입은 상해 를 말하며, 유독 가스 또는 유독 물질을 반려묘가 '
 '우연히 일시적으로 흡입, 흡수\n'
 '또는 섭취한 결과로 생긴 중독 증상을 포함합니다. 그러나 음식물 섭취로 인한 증 상, 세균성 음식물 중독과 상습적으로 흡입, 흡수 또는 '
 '섭취한 결과로 생긴 중독 증상은 포함되지 않습니다.\n'
 '<용어풀이>\n'
 '[음식물]'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 96},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000525',
              'chunk_char_len': 223,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
