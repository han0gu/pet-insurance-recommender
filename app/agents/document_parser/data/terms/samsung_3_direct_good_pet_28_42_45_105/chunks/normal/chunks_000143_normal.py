from langchain_core.documents import Document

chunk = Document(
    page_content=('를 제기하여 계약자, 피보험자 또는 보험수익자에게 손해를 가한 경우에는 그에 따른 손해를 배상할 책임을 집니다.\n'
 '③ 회사가 보험금 지급여부 및 지급금액에 관하여 현저하게 공정을 잃은 합의로 보험수 익자에게 손해를 가한 경우에도 회사는 제2항에 따라 '
 '손해를 배상할 책임을 집니다.\n'
 '<용어풀이>\n'
 '[현저하게 공정을 잃은 합의]\n'
 '회사가 보험수익자의 경제적. 신체적. 정신적인 어려움, 경솔함, 경험 부족 등을 이용하여 동일. 유사 사례에 비추어 보험수익자에게 매우 '
 '불합리하게 합의를 하는 것을 의미합니다.\n'
 '제42조 (개인정보보호)'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 67,
         'page': 42},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000143',
              'chunk_char_len': 292,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
