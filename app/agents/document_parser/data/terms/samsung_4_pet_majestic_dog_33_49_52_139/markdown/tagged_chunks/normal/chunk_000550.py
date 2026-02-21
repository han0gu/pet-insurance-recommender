from langchain_core.documents import Document

chunk = Document(
    page_content=('제외한 금액을 아래에 정한 한도로 제5항에 따라 보험수익자에게 반려견의료비확대보\n'
 '장 보험금으로 보상하여 드립니다.| 지급기준 | 1회당 보상한도액 |\n'
 '| --- | --- |\n'
 '| 이물제거를 목적으로 내시경을 받은 경우 | 200만원 |\n'
 '| 이물제거를 목적으로 구토유도약물을 투약한 경우 | 20만원 |\n'
 '- ② 이물제거(내시경)과 이물제거(구토유도약물)을 동일한 날에 받은 경우 이물제거(내시\n'
 '- 경) 보험금만 지급됩니다.\n'
 '- ③ 제1항에서 정한「반려견의료비(치과및구강질환포함)(수술당일제외,검사비포함)보험금'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'limit', 'risk_domains': ['dental', 'digestive']},
 'indexing': {'chunk_id': 'chunk_000550',
              'chunk_char_len': 283,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
