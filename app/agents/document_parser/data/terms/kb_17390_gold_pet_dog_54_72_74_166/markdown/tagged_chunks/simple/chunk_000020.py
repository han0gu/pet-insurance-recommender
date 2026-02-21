from langchain_core.documents import Document

chunk = Document(
    page_content=('- 지급사유가 발생한 경우에는 보장합니다)\n'
 '56 KB 금쪽같은 펫보험(강아지)(무배당)(26.01)# 3. 선박에 탑승하는 것을 직무로하는 사람이 직무상 선박에 탑승하고 있는 '
 '동안제6조(보험금 지급사유의 통지)계약자 또는 피보험자나 보험수익자는 제3조(보험금의 지급사유)에서 정한 보험금 지급사유의 발생을 안 '
 '때에는 지체없이 그 사실을 회사에 알려야 합니다.제7조(보험금의 청구)# \uf000- 보험수익자는 다음의 서류를 제출하고 보험금을 '
 '청구하여야 합니다.\n'
 '- 1. 청구서(회사 양식)'),
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
 'indexing': {'chunk_id': 'chunk_000020',
              'chunk_char_len': 268,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
