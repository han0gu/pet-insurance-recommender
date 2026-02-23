from langchain_core.documents import Document

chunk = Document(
    page_content=('- 결하여야 합니다.\n'
 '# 제2조(상법 제735조3의 적용)- ① 제1조의 단체가 피보험자를 확정할 수 있고 계약의 일괄적 관리가 가능하며, 규약에 따라 '
 '계약을\n'
 '- 체결하는 경우 피보험자의 서면에 의한 동의를 얻지 않아도 되며, 계약자에게만 보험증권을 드릴\n'
 '- 수 있습니다.\n'
 '- ② 제1항의 규약은 보험의 종류 및 일괄 가입에 관한 사항이 포함되어야 하며, 동의 또는 협의를 통하\n'
 '- 여 피보험자들의 의사가 규약에 반영될 수 있어야 합니다. 다만, 보험수익자를 계약자 등 피보험자'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000108',
              'chunk_char_len': 267,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
