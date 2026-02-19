from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 보험수익자를 계약자 등 피보험자 의 이해에 반하는 자로 지정하는 경우에는 해당 내용이 규약에 반영되어야 하며, 반영되지 않은 '
 '경 우에는 별도 피보험자의 동의를 받아야 합니다. ③ 회사는 계약자를 통해 단체의 규약이 제2항을 충족하고 있는 지 확인을 해야 하며, '
 '계약자는 이에 협조하여야 합니다.'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 35},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000176',
              'chunk_char_len': 170,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
