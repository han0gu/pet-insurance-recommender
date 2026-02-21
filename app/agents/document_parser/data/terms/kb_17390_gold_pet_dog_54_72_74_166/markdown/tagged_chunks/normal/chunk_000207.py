from langchain_core.documents import Document

chunk = Document(
    page_content=('용 어 풀 이 보험안내자료- \n'
 '| 계약의 청약을 권유하기 | 위하여 만든 자료 등을 말합니다. |\n'
 '| --- | --- |\n'
 '- 70 -- \n'
 '- 제48조(회사의 손해배상책임)\n'
 '- \uf000 회사는 계약과 관련하여 임직원, 보험설계사 및 대리점의 책임있는 사유로 계약자,\n'
 '- 피보험자 및 보험수익자에게 발생된 손해에 대하여 관계 법령 등에 따라 손해배상\n'
 '- 의 책임을 집니다.\n'
 '- \uf000 회사는 보험금 지급 거절 및 지연지급의 사유가 없음을 알았거나 알 수 있었는데도\n'
 '- 소를 제기하여 계약자, 피보험자 또는 보험수익자에게 손해를 가한 경우에는 그에'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000207',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
