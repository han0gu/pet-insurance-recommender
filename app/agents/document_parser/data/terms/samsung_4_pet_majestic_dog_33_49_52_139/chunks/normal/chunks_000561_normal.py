from langchain_core.documents import Document

chunk = Document(
    page_content=('렙토스피라 감염증 | 랩토스피라 세균에 감염되어 황달, 신부전 등의 증상을 일으킴\n'
 '심장사상충 감염증 | 개사상충(기생충)이 심장이나 폐혈관에서 기생하며 호흡곤란, 혈액순환 장애 등의 증상을 일으킴\n'
 '켄넬코프 | 전염성 기관지염으로 기침이나 발열 등 사람이 걸리는 감기와 비슷한 증상\n'
 '잔존유치 | 영구치가 났는데도 불구하고 유치가 남아있어서 발치를 하는 경우\n'
 '잠복고환 | 고환이 음낭까지 내려오지 못하는 증상'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 101},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000561',
              'chunk_char_len': 226,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
