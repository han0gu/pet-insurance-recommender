from langchain_core.documents import Document

chunk = Document(
    page_content=('보험료 자동이체 특별약관\n'
 '제1 조(보험료납입)\n'
 '계약자는 제2회 이후의 보험료부터 이 특별약관에 따라 계약자의 지정계좌를 이용하여 보험료를 자동 납입 합니다.\n'
 '제2조(보험료의 영수)\n'
 '자동납입일자는 이 청약서에 기재된 보험료납입 해당일에도 불구하고 회사와 계약자가 별도로 약정한 일자로 합니다.\n'
 '제3조(계약 후 알릴 의무)\n'
 '계약자는 지정계좌의 번호가 변경 또는 거래정지된 경우에는 그 사실을 즉시 회사에 알려야 합니다.\n'
 '제4조(준용규정)\n'
 '이 특별약관에 정하지 않은 사항은 보통약관을 따릅니다.'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 35,
         'page': 25},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000124',
              'chunk_char_len': 273,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
