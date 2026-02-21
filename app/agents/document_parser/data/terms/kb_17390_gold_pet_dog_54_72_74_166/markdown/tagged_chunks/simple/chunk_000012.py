from langchain_core.documents import Document

chunk = Document(
    page_content=('- 1. 일요일\n'
 '- 2. 국경일 중 3 ‧ 1절, 광복절, 개천절 및 한글날\n'
 '- 3. 1월 1일\n'
 '- 4. 설날 전날, 설날, 설날 다음날 (음력 12월 말일, 1월 1일, 2일)\n'
 '- 5. 삭제 <2005. 6. 30>\n'
 '- 6. 부처님오신날 (음력 4월 8일)\n'
 '- 7. 5월 5일 (어린이날)\n'
 '- 8. 6월 6일 (현충일)\n'
 '9.(음력 8월 14일, 15일, 16일)추석 전날, 추석, 추석 다음날10. 12월 25일(기독탄신일)\n'
 '10의2. 「공직선거법」제34조에 따른 임기만료에 의한 선거의 선거일'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000012',
              'chunk_char_len': 277,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
