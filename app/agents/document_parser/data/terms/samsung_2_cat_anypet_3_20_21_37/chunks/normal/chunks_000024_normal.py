from langchain_core.documents import Document

chunk = Document(
    page_content=('제6조(손해의 통지 및 조사)\n'
 '① 계약자 또는 피보험자는 제4조(보상하는 손해)에서 정한 사고가 생긴 것을 안 때에는 지체없이 그 사실을 회사에 알려야 합니다. ② '
 '계약자 또는 피보험자가 제1항의 통지를 게을리하여 손해가 증가된 때에는 회사는 그 증가된 손해 는 보상하여 드리지 않습니다. ③ 회사가 '
 '위 제1항에 대한 손해의 사실을 확인하기 어려운 경우에는 계약자 또는 피보험자에게 필요 한 증거자료의 제출을 요구할 수 있습니다.\n'
 '제7조(보험금의 청구)\n'
 '① 피보험자가 보험금을 청구할 때에는 다음의 서류를 회사에 제출하여야 합니다.'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 35,
         'page': 7},
 'term_type': 'basic',
 'clause': {'clause_type': 'claim', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000024',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
