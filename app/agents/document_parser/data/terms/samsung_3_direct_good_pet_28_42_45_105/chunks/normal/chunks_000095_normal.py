from langchain_core.documents import Document

chunk = Document(
    page_content=('<예시안내>\n'
 '[보험나이 계산]\n'
 '생년월일 : 1988년 10월 2일\n'
 '예1) 계 약 일 : 2021년 3월 13일\n'
 '⇒ 2021년 3월 13일 - 1988년 10월 2일 32년 5개월 11일 = 32세\n'
 '예2) 계 약 일 : 2021년 4월 13일\n'
 '⇒ 2021년 4월 13일 - 1988년 10월 2일 32년 6개월 11일 = 33세\n'
 '[계약해당일 계산]'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 67,
         'page': 37},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000095',
              'chunk_char_len': 193,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
