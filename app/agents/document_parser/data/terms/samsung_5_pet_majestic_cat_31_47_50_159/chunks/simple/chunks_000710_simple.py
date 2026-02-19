from langchain_core.documents import Document

chunk = Document(
    page_content=('제3조 (이물제거(내시경) 및 이물제거(구토유발약물)의 정의)\n'
 '① 이 특별약관에서 「이물제거(내시경)」 이라 함은 반려동물의 위장 등 내부의 이물질을 제거하기 위하여 수술을 동반하지 않고 내시경 및 '
 '내시경포셉을 이용하여 비침습적으 로 시행하는 의료행위를 말합니다. ② 이 특별약관에서 「이물제거(구토유발약물)」 이라 함은 반려동물의 '
 '위장 등 내부의 이 물질을 제거하기 위하여 수술 및 내시경을 동반하지 않고 구토유발을 목적으로 한 약 물을 이용한 의료행위를 말합니다.\n'
 '제4조 (보험금을 지급하지 않는 사유)'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 114},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000710',
              'chunk_char_len': 283,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
