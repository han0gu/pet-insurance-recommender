from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 보험계약<br>자가 보험수익자를 피보험자 또는 그 상속인이 아닌 자로 지정하는 경우에는 해당 내<br>용이 규약에 반영되어야 '
 '하며, 반영되지 않은 경우에는 별도 피보험자의 동의를 받아야<br>합니다.<br>③ 보험회사는 계약자를 통해 단체의 규약이 제2항을 '
 "충족하고 있는 지 확인을 해야 하며,<br>계약자는 이에 협조하여야 합니다.</p><h1 id='61' "
 "style='font-size:14px'>제3조(단체요율의 적용)</h1><br><p id='62' "
 "data-category='paragraph'"),
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
 'indexing': {'chunk_id': 'chunk_000315',
              'chunk_char_len': 289,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
