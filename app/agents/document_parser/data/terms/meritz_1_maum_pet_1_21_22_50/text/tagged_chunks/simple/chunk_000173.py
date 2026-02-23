from langchain_core.documents import Document

chunk = Document(
    page_content=('리가 가능하며, 규약에 따라 계약을 체결하는 경우 피보험자의 서면에 의한 동의를 얻지\n'
 '않아도 되며, 계약자에게만 보험증권을 드릴 수 있습니다.\n'
 '② 제1항의 규약은 보험의 종류 및 일괄 가입에 관한 사항이 포함되어야 하며, 동의 또는\n'
 '협의를 통하여 피보험자들의 의사가 규약에 반영될 수 있어야 합니다. 다만, 보험계약\n'
 '자가 보험수익자를 피보험자 또는 그 상속인이 아닌 자로 지정하는 경우에는 해당 내\n'
 '용이 규약에 반영되어야 하며, 반영되지 않은 경우에는 별도 피보험자의 동의를 받아야\n'
 '합니다.'),
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
 'indexing': {'chunk_id': 'chunk_000173',
              'chunk_char_len': 274,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
