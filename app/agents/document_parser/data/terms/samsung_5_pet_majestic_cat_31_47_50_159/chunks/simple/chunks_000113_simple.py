from langchain_core.documents import Document

chunk = Document(
    page_content=('<예시안내>\n'
 '[보험나이 계산]\n'
 '생년월일 : 1988년 10월 2일\n'
 '예1) 계 약 일 : 2022년 3월 13일\n'
 '⇒ 2022년 3월 13일 - 1988년 10월 2일 33년 5개월 11일 = 33세\n'
 '[계약해당일 계산]\n'
 '최초계약일과 동일한 월, 일을 말합니다. 계약일 : 2022년 4월 10일 ⇒ 계약해당일 : 매년 4월 10일 단 , 계약해당일 2월 '
 '29일이 없을 경우에는 2월 28일을 계약해당일로 합니다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 41},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000113',
              'chunk_char_len': 228,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
