from langchain_core.documents import Document

chunk = Document(
    page_content=('일자로 합니다.# 제3조(계약 후 알릴 의무)계약자는 지정계좌의 번호가 변경 또는 거래정지된 경우에는 그 사실을 즉시 회사에 알려야 '
 '합니다.# 제4조(준용규정)이 특별약관에 정하지 않은 사항은 보통약관을 따릅니다.33당신에게 좋은보험 삼성화재# 초회보험료자동이체 '
 '특별약관# 제1 조(보험료의 납입)- 계약자 또는 피보험자는 제1회 보험료의 납입방법을 거래은행 지정계좌를 통한 자동납입으로 가입\n'
 '- 하고자 하는 경우에, 회사는 청약서를 접수하고 자동이체신청에 필요한 정보를 제공한 때(다만, 계'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000137',
              'chunk_char_len': 275,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
