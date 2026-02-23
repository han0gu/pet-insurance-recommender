from langchain_core.documents import Document

chunk = Document(
    page_content=('- ② 제1항의 보험나이는 계약일 현재 피보험자의 실제 만 나이를 기준으로 6개월 미만의\n'
 '- 끝수는 버리고 6개월 이상의 끝수는 1년으로 하여 계산하며, 이후 매년 계약해당일에\n'
 '- 나이가 증가하는 것으로 합니다.\n'
 '- ③ 피보험자의 나이 또는 성별에 관한 기재사항이 사실과 다른 경우에는 정정된 나이 또\n'
 '- 는 성별에 해당하는 보험금 및 보험료로 변경합니다.\n'
 '# <예시안내>[보험나이 계산]생년월일 : 1988년 10월 2일예1) 계 약 일 : 2022년 3월 13일⇒ 2022년 3월 13일\n'
 '- 1988년 10월 2일'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000099',
              'chunk_char_len': 289,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
