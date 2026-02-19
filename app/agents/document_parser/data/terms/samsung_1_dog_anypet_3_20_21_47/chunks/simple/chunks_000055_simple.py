from langchain_core.documents import Document

chunk = Document(
    page_content=('- 보험금 지급제한 사유 및 지급절차 - 청약의 철회에 관한 사항 - 계약의 해지 및 해제 - 분쟁조정 절차에 관한 사항 - '
 '예금자보호법에 따른 보호여부 - 환급금에 관한 사항 - 고지의무 및 통지의무 위반의 효과 - 만기시 자동갱신되는 보험계약의 경우 '
 '자동갱신의 조건 - 그 밖에 약관에 기재된 보험계약의 중요사항\n'
 '② 제1항과 관련하여 통신판매계약의 경우, 회사는 계약자가 가입한 특약만 포함한 약관을 드리며, 전 화를 이용하여 체결하는 계약은 '
 '계약자의 동의를 얻어 다음의 방법으로 약관의 중요한 내용을 설 명할 수 있습니다.'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 12},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000055',
              'chunk_char_len': 295,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
