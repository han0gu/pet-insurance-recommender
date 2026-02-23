from langchain_core.documents import Document

chunk = Document(
    page_content=('자동납입일자는 이 청약서에 기재된 보험료 납입해당일에도 불구하고 매월 회사가 정하는 날 중 계약자가 희망하는 일자로 합니다.제3조(계약 '
 '후 알릴 의무)\n'
 '계약자는 지정계좌의 번호가 변경 또는 거래 정지된 경우에는 이 사실을 즉시 회사에# 알려야 합니다.제4조(준용규정)# 이 특별약관에 '
 '정하지 않은 사항은 보통약관 및 해당 특별약관을 따릅니다.| 3-1. 초회보험료자동납입 추가특별약관 |\n'
 '| --- |'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000770',
              'chunk_char_len': 224,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
