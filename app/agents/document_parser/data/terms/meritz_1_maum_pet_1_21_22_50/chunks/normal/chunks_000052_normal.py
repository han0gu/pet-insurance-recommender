from langchain_core.documents import Document

chunk = Document(
    page_content=('제14조(대표자의 지정)\n'
 '① 계약자 또는 보험수익자가 2명 이상인 경우에는 각 대표자를 1명 지정하여야 합니다. 이 경우 그 대표자는 각각 다른 계약자 또는 '
 '보험수익자를 대리하는 것으로 합니다. ② 지정된 계약자 또는 보험수익자의 소재가 확실하지 않은 경우에는 이 계약에 관하여 회 사가 계약자 '
 '또는 보험수익자 1명에 대하여 한 행위는 각각 다른 계약자 또는 보험수 익자에게도 효력이 미칩니다. ③ 계약자가 2명 이상인 경우에는 그 '
 '책임을 연대로 합니다.\n'
 '【계약자가 2명 이상인 경우】'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 9},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000052',
              'chunk_char_len': 271,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
