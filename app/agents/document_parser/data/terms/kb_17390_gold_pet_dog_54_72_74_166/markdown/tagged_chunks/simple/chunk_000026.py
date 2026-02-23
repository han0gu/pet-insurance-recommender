from langchain_core.documents import Document

chunk = Document(
    page_content=('험수익자의 책임있는 사유로 보험금 지급사유의 조사와 확인이 지연되는 경우| 6. 보험금 | 지급사유에 대해 제3자의 의견에 따르기로 한 '
 '경우 |\n'
 '| --- | --- |\n'
 '| 용 어 풀 이 분쟁조정 신청 분쟁조정 신청은 이 약관의 ｢분쟁의 조정｣ 조항에 따르며 분쟁조정 신청 대상기 | 용 어 풀 이 분쟁조정 '
 '신청 분쟁조정 신청은 이 약관의 ｢분쟁의 조정｣ 조항에 따르며 분쟁조정 신청 대상기 |\n'
 '- 관은 금융감독원의 금융분쟁조정위원회를 말합니다.\n'
 '- \uf000 제2항에 의하여 장해지급률의 판정 및 지급할 보험금의 결정과 관련하여 확정된 장'),
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
 'indexing': {'chunk_id': 'chunk_000026',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
