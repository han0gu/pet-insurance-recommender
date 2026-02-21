from langchain_core.documents import Document

chunk = Document(
    page_content=('- 난 때에 계약자 또는 보험수익자에게 도달된 것으로 봅니다.\n'
 '- 8 -# 제13조(보험수익자의 지정)# 보험수익자를 지정하지 않은 때에는 보험수익자를 피보험자로 합니다.# 제14조(대표자의 지정)- '
 '① 계약자 또는 보험수익자가 2명 이상인 경우에는 각 대표자를 1명 지정하여야 합니다.\n'
 '- 이 경우 그 대표자는 각각 다른 계약자 또는 보험수익자를 대리하는 것으로 합니다.\n'
 '- ② 지정된 계약자 또는 보험수익자의 소재가 확실하지 않은 경우에는 이 계약에 관하여 회'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000050',
              'chunk_char_len': 259,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
