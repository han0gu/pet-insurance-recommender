from langchain_core.documents import Document

chunk = Document(
    page_content=('이 경우 그 대표자는 각각 다른 계약자 또는 보험수익자를 대리하는 것으로 합니다.\n'
 '② 지정된 계약자 또는 보험수익자의 소재가 확실하지 않은 경우에는 이 계약에 관하여 회\n'
 '사가 계약자 또는 보험수익자 1명에 대하여 한 행위는 각각 다른 계약자 또는 보험수\n'
 '익자에게도 효력이 미칩니다.\n'
 '③ 계약자가 2명 이상인 경우에는 그 책임을 연대로 합니다.【계약자가 2명 이상인 경우】계약자가 2명 이상인 경우, 보험료 납입의무 등 '
 '보험계약에 따른 계약자의 의무를'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000046',
              'chunk_char_len': 251,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
