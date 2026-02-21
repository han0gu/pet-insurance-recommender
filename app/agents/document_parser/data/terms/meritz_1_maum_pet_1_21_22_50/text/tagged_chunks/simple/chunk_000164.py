from langchain_core.documents import Document

chunk = Document(
    page_content=('치료비 보험금을 보상하여 드리지 않습니다.제2조(준용규정)이 특별약관에 정하지 않은 사항은 보통약관을 따릅니다.- 33 -보험료분납 '
 '특별약관∥제1조(보험료의 분납)계약자는 이 특별약관에 따라 보험료를 ( )회에 분할하여 회사에 납입합니다.제2조(나눠 내는 보험료의 '
 '납입)① 계약자는 계약을 체결할 때에 제1회 나눠 내는 보험료를 납입하고 제2회 이후의 나눠\n'
 '내는 보험료는 아래에 기재된 납입기일까지 납입하여야 합니다.( )회 분납: 제 1회: 계약의 청약일 (총 보험료의 ( )% 해당액)'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000164',
              'chunk_char_len': 273,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
