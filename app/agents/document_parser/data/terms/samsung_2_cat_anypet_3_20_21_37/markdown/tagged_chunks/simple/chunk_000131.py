from langchain_core.documents import Document

chunk = Document(
    page_content=('- 을 통보합니다.\n'
 '# - 매주 □, 매월 □, 기타 □( )# 제4조(보험료 정산기간)계약자는 다음 중 어느 하나의 것으로 보험료를 정산하기로 약정하고, '
 '이 기간을 보험료정산기간 (이\n'
 '하 「정산기간」 이라 합니다)이라 합니다.# 1. 계약 기간 중- 매월 □, - 매6개월 □, - 기타 □ ()# 2. 보험기간 종료 후 '
 '□# 제5조(예치보험료)계약자는 제4조(보험료 정산기간)의 매 정산기간이 시작될 때 마다 정산기간 동안의 예상 피보험자 수에'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000131',
              'chunk_char_len': 250,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
