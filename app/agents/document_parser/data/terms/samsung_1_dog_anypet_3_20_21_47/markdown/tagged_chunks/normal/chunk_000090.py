from langchain_core.documents import Document

chunk = Document(
    page_content=('- 기하여 계약자 또는 피보험자에게 손해를 가한 경우에는 그에 따른 손해를 배상할 책임을 집니다.\n'
 '- ③ 회사가 보험금 지급여부 및 지급금액에 관하여 현저하게 공정을 잃은 합의로 계약자 또는 피보험\n'
 '- 자에게 손해를 가한 경우에도 회사는 제2항에 따라 손해를 배상할 책임을 집니다.\n'
 '- 19 -당신에게 좋은보험 삼성화재【현저하게 공정을 잃은 합의】 회사가 계약자 또는 피보험자의 경제적. 신체적. 정신적인 어려움, '
 '경솔함,\n'
 '경험 부족 등을 이용하여 동일·유사 사례에 비추어 계약자 또는 피보험자에게 매우 불합리하게 합의를 하'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000090',
              'chunk_char_len': 293,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
