from langchain_core.documents import Document

chunk = Document(
    page_content=('③ 제2항에서 정한 조치에 다른 진료를 병행하여 실시한 경우에는 제2항에서 정한 조치(\n'
 '마취 비용을 포함합니다)에 대한 보험금은 지급하지 않습니다.# 제 4조 (이물제거(내시경) 및 이물제거(구토유발약물)의 정의)① 이 '
 '특별약관에서「이물제거(내시경)」란 반려견의 위장 등 내부의 이물질을 제거하기\n'
 '위하여 수술을 동반하지 않고 내시경 및 내시경포셉을 이용하여 비침습적으로 시행하\n'
 '는 의료행위를 말합니다.\n'
 '② 이 특별약관에서「이물제거(구토유발약물)」란 반려견의 위장 등 내부의 이물질을 제'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000568',
              'chunk_char_len': 271,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
