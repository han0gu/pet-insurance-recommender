from langchain_core.documents import Document

chunk = Document(
    page_content=('보험료 자동납입 특별약관\n'
 '제1조(보험료의 납입)\n'
 '계약자는 보험료 분납 특별약관에 의하여 제2회 이후의 보험료부터 이 특별약관에 따라 계약자의 거래은행 지정계좌를 이용하여 보험료를 자동 '
 '납입합니다.\n'
 '제2조(자동납입 신청)\n'
 '계약자는 보험계약과 동시에 계약자의 거래은행 지정계좌를 이용하여 보험료를 자동 납입 하는 별첨 신청서를 작성합니다.\n'
 '제3조(보험료의 영수)\n'
 '자동납입일자는 이 보험계약 청약서에 기재된 보험료 납입 해당일에도 불구하고 매월 회사 가 정하는 날 중 보험계약자가 희망하는 일자로 '
 '합니다.\n'
 '제4조(계약 후 알릴 의무)'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 35},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000197',
              'chunk_char_len': 295,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
