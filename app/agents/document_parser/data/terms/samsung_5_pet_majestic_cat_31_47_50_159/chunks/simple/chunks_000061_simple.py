from langchain_core.documents import Document

chunk = Document(
    page_content=('<용어풀이>\n'
 '[직업] 1) 생계유지 등을 위하여 일정한 기간동안(예: 6개월 이상) 계속하여 종사하는 일 2) 1)에 해당하지 않는 경우에는 개인의 '
 '사회적 신분에 따르는 위치나 자리를 말함 예 ) 학생, 미취학아동, 무직 등\n'
 '[직무] 직책이나 직업상 책임을 지고 담당하여 맡은 일\n'
 '2. 보험증권 등에 기재된 피보험자의 운전 목적이 변경된 경우'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 36},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000061',
              'chunk_char_len': 191,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
