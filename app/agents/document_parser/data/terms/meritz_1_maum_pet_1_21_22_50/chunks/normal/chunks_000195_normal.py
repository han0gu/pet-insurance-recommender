from langchain_core.documents import Document

chunk = Document(
    page_content=('보험료분납 특별약관∥\n'
 '제1조(보험료의 분납)\n'
 '계약자는 이 특별약관에 따라 보험료를 ( )회에 분할하여 회사에 납입합니다.\n'
 '제2조(나눠 내는 보험료의 납입)\n'
 '① 계약자는 계약을 체결할 때에 제1회 나눠 내는 보험료를 납입하고 제2회 이후의 나눠 내는 보험료는 아래에 기재된 납입기일까지 '
 '납입하여야 합니다.\n'
 '( )회 분납: 제 1회: 계약의 청약일 (총 보험료의 ( )% 해당액) 제( )회: 년 월 일 (총 보험료의 ( )% 해당액)'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 34},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000195',
              'chunk_char_len': 241,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
