from langchain_core.documents import Document

chunk = Document(
    page_content=('【약관의 중요한 내용】 금융소비자 보호에 관한 법률 제19조(설명의무), 동법 시행령 제13조, 금융소 비자 보호에 관한 감독규정 '
 '[별표3]에서 정한 다음의 내용을 포함합니다.\n'
 '- 보험금 지급제한 사유 및 지급절차 - 청약의 철회에 관한 사항 - 계약의 해지 및 해제 - 분쟁조정 절차에 관한 사항 - '
 '예금자보호법에 따른 보호여부 - 환급금에 관한 사항 - 고지의무 및 통지의무 위반의 효과 - 만기시 자동갱신되는 보험계약의 경우 '
 '자동갱신의 조건 - 그 밖에 약관에 기재된 보험계약의 중요사항'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 35,
         'page': 12},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000051',
              'chunk_char_len': 275,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
