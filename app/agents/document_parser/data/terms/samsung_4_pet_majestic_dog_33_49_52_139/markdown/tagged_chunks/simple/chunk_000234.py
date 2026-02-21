from langchain_core.documents import Document

chunk = Document(
    page_content=('- 는 성별에 해당하는 보험금 및 보험료로 변경합니다.\n'
 '<예시안내>[보험나이 계산]생년월일 : 1988년 10월 2일예1) 계 약 일 : 2022년 3월 13일⇒ 2022년 3월 13일\n'
 '- 1988년 10월 2일\n'
 '33년 5개월 11일 = 33세예 2) 계 약 일 : 2022년 4월 13일[계약해당일 계산]\n'
 '최초계약일과 동일한 월, 일을 말합니다.⇒ 2022년 4월 13일\n'
 '- 1988년 10월 2일\n'
 '33년 6개월 11일 = 34세- \n'
 '- 60 -계약일 : 2022년 4월 10일 ⇒ 계약해당일 : 매년 4월 10일'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000234',
              'chunk_char_len': 288,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
