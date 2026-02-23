from langchain_core.documents import Document

chunk = Document(
    page_content=('- \uf000 회사는 보험금 지급 거절 및 지연지급의 사유가 없음을 알았거나 알 수 있었는데\n'
 '- 도 소를 제기하여 계약자, 피보험자 또는 보험수익자에게 손해를 가한 경우에는\n'
 '- 그에 따른 손해를 배상할 책임을 집니다.\n'
 '- \uf000 회사가 보험금 지급여부 및 지급금액에 관하여 현저하게 공정을 잃은 합의로 보\n'
 '- 험수익자에게 손해를 가한 경우에도 회사는 제2항에 따라 손해를 배상할 책임을\n'
 '- 집니다.\n'
 '| 용 어 풀 | 이 현저하게 공정을 잃은 합의 |\n'
 '| --- | --- |'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000531',
              'chunk_char_len': 259,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
