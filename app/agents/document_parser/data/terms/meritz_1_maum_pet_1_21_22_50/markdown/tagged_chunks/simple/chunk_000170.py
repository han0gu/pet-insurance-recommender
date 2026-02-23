from langchain_core.documents import Document

chunk = Document(
    page_content=('내는 보험료는 아래에 기재된 납입기일까지 납입하여야 합니다.( )회 분납: 제 1회: 계약의 청약일 (총 보험료의 ( )% 해당액)제( '
 ')회: 년 월 일 (총 보험료의 ( )% 해당액)- ② 보험기간이 시작된 후라도 제1항의 제1회 나눠 내는 보험료를 납입하기 전에 생긴 '
 '사\n'
 '- 고는 보상하여 드리지 않습니다.\n'
 '- ③ 보험기간동안 이 계약의 보험요율이 변경된 경우라도 이 특별약관에 따라 납입하는 분\n'
 '- 납보험료는 변경적용하지 않습니다. 다만, 보통약관 제16조(계약 후 알릴 의무)에 따라\n'
 '- 보험료가 변경된 경우에는 예외로 합니다.'),
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
 'indexing': {'chunk_id': 'chunk_000170',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
