from langchain_core.documents import Document

chunk = Document(
    page_content=('기재된 반려동물에게 시행한 치료로서 다음 각 호의 사항을 말합니다.- 1. 상해 또는 질병의 직접적인 치료를 목적으로 "MRI/CT"를 '
 '받은 경우 상\n'
 '- 해\n'
 '- 2. 백내장 또는 녹내장의 직접적인 치료를 목적으로 "백내장/녹내장수술"을 받은\n'
 '- 및\n'
 '- 경우\n'
 '- 질\n'
 '- 3. 이물 섭취 치료를 직접적인 목적으로 치료 중 "이물제거(내시경)" 또는 "이물\n'
 '- 병\n'
 '- 제거(구토유도약물)"를 받은 경우\n'
 '- 4. 상해로 인한 창상 또는 교상의 직접적인 치료를 목적으로 "창상/교상치료"를\n'
 '- 받은 경우\n'
 '- 반'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'eye']},
 'indexing': {'chunk_id': 'chunk_000589',
              'chunk_char_len': 280,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
