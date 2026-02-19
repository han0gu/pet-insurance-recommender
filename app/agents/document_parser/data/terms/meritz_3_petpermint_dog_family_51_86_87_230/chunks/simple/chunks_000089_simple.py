from langchain_core.documents import Document

chunk = Document(
    page_content=('【약관의 중요한 내용 예시】\n'
 '- 청약의 철회에 관한 사항 - 지급한도, 면책사항, 감액지급 사항 등 보험금 지급제 한 조건 - 계약 전 알릴 의무(고지의무) 위반의 '
 '효과 - 계약의 취소 및 무효에 관한 사항 - 해약환급금에 관한 사항 - 분쟁조정절차에 관한 사항 - 만기시 자동갱신되는 보험계약의 경우 '
 '자동갱신의 조건 - 저축성 보험계약의 공시이율 - 유배당 보험계약의 경우 계약자 배당에 관한 사항 - 그 밖에 약관에 기재된 보험계약의 '
 '중요사항'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 69},
 'term_type': 'basic',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000089',
              'chunk_char_len': 249,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
