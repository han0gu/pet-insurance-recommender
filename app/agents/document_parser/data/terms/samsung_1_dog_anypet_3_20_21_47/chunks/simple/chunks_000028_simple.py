from langchain_core.documents import Document

chunk = Document(
    page_content=('② 계약자 또는 피보험자가 제1항의 통지를 게을리하여 손해가 증가된 때에는 회사는 그 증가된 손해 는 보상하여 드리지 않습니다. ③ '
 '회사가 위 제1항에 대한 손해의 사실을 확인하기 어려운 경우에는 계약자 또는 피보험자에게 필요 한 증거자료의 제출을 요구할 수 '
 '있습니다.\n'
 '제7조(보험금의 청구)\n'
 '① 피보험자가 보험금을 청구할 때에는 다음의 서류를 회사에 제출하여야 합니다.'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 8},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000028',
              'chunk_char_len': 207,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
