from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 보험계약 자가 보험수익자를 피보험자 또는 그 상속인이 아닌 자로 지정하는 경우에는 해당 내 용이 규약에 반영되어야 하며, '
 '반영되지 않은 경우에는 별도 피보험자의 동의를 받아야 합니다. ③ 보험회사는 계약자를 통해 단체의 규약이 제2항을 충족하고 있는 지 '
 '확인을 해야 하며, 계약자는 이에 협조하여야 합니다.'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 37},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000207',
              'chunk_char_len': 177,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
