from langchain_core.documents import Document

chunk = Document(
    page_content=('- 를 제기하여 계약자, 피보험자 또는 보험수익자에게 손해를 가한 경우에는 그에 따른\n'
 '- 20 -# 손해를 배상할 책임을 집니다.③ 회사가 보험금 지급여부 및 지급금액에 관하여 현저하게 공정을 잃은 합의로 보험수익\n'
 '자에게 손해를 가한 경우에도 회사는 제2항에 따라 손해를 배상할 책임을 집니다.# 【현저하게 공정을 잃은 합의】사회통념상 일반 '
 '보통인이라면 그 같은 일을 하지 않을 정도로 현저하게 공정성을'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000116',
              'chunk_char_len': 225,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
