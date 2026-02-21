from langchain_core.documents import Document

chunk = Document(
    page_content=('등<br>- "이물제거(구토유도약물)" 시행한 경우 : 구토유도약물 처방이 명시된 동물 상<br>병원 진료비 영수증(치료비 세부내역 '
 '포함) 및 수의사처방전 해<br>- "특정약물치료Ⅱ" 시행한 경우 : 특정약물치료Ⅱ에 해당하는 약물이 명시된<br>수의사처방전<br>- '
 '"특정재활치료Ⅱ" 시행한 경우 : 특정재활치료Ⅱ에 해당하는 치료명이 명시<br>된 치료비 세부내역 포함<br>- "항암약물치료" 시행한 '
 "경우 : 항암약물치료에 해당하는 약물이 명시된 수<br>질<br>의사처방전</p><br><p id='27'"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_001055',
              'chunk_char_len': 285,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
