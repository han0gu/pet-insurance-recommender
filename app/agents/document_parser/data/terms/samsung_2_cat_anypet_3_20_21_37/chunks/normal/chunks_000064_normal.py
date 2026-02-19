from langchain_core.documents import Document

chunk = Document(
    page_content=('제22조(제2회 이후 보험료의 납입)\n'
 '계약자는 제2회 이후의 보험료를 납입기일까지 납입하여야 하며, 회사는 계약자가 보험료를 납입한 경 우에는 영수증을 발행하여 드립니다. '
 '다만, 금융회사(우체국을 포함합니다)를 통하여 보험료를 납입한 경우에는 그 금융회사 발행 증빙서류를 영수증으로 대신합니다.\n'
 '【납입기일】 계약자가 제2회 이후의 보험료를 납입하기로 한 날을 말합니다.\n'
 '제23조[보험료의 납입이 연체되는 경우 납입최고(독촉)와 계약의 해지]'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 35,
         'page': 14},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000064',
              'chunk_char_len': 246,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
