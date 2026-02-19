from langchain_core.documents import Document

chunk = Document(
    page_content=('보험료분납 특별약관\n'
 '제1조(보험료의 납입)\n'
 '① 이 특별약관에 따라 계약자는 보험기간이 1년인 보험 계약에 대하여 보험료를 제2항에 정한 바에 따라 나누어 납입할 수 있습니다. ② '
 '계약자는 이 보험의 보험료 및 해약환급금 산출방법서에서 정한 방법에 의하여 계산된 분납보험료 를 해당 보험기간 및 분할회수에 따라 아래에 '
 '정한 시기까지 납입하여야 합니다.\n'
 '보험 기간 | 제2회 이후 분납보험료 납입시기\n'
 '분할회수 | 이후 분납보험료 제2회'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 32},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000164',
              'chunk_char_len': 241,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
