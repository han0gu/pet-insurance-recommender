from langchain_core.documents import Document

chunk = Document(
    page_content=('제13조(보험수익자의 지정)\n'
 '보험수익자를 지정하지 않은 때에는 보험수익자를 만기환급 금의 경우는 계약자로 하고, 사망보험금의 경우는 피보험자 의 법정상속인, 이 외의 '
 '보험금은 피보험자로 합니다.\n'
 '【법정상속인】\n'
 '법정상속인이라 함은 피상속인의 사망에 의하여 민법의 규정에 의한 상속순위에 따라 상속받는 자를 말합니다.\n'
 '제14조(대표자의 지정)'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 57},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000047',
              'chunk_char_len': 189,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
