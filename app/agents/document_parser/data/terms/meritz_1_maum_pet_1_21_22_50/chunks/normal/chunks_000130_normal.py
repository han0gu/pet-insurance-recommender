from langchain_core.documents import Document

chunk = Document(
    page_content=('제39조(회사의 손해배상책임)\n'
 '① 회사는 계약과 관련하여 임직원, 보험설계사 및 대리점의 책임있는 사유로 계약자, 피 보험자 및 보험수익자에게 발생된 손해에 대하여 '
 '관계 법령 등에 따라 손해배상의 책 임을 집니다. ② 회사는 보험금 지급 거절 및 지연지급의 사유가 없음을 알았거나 알 수 있었는데도 소 '
 '를 제기하여 계약자, 피보험자 또는 보험수익자에게 손해를 가한 경우에는 그에 따른'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 20},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000130',
              'chunk_char_len': 214,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
