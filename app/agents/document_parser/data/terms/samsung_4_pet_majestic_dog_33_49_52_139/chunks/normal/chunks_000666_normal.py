from langchain_core.documents import Document

chunk = Document(
    page_content=('제 4조 (이물제거(내시경) 및 이물제거(구토유발약물)의 정의)\n'
 '① 이 특별약관에서「이물제거(내시경)」란 반려견의 위장 등 내부의 이물질을 제거하기 위하여 수술을 동반하지 않고 내시경 및 내시경포셉을 '
 '이용하여 비침습적으로 시행하 는 의료행위를 말합니다. ② 이 특별약관에서「이물제거(구토유발약물)」란 반려견의 위장 등 내부의 이물질을 제 '
 '거하기 위하여 수술 또는 내시경을 동반하지 않고 구토유발을 목적으로 한 약물을 이 용한 의료행위를 말합니다.\n'
 '<유의사항>\n'
 '[수술]'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 112},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000666',
              'chunk_char_len': 261,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
