from langchain_core.documents import Document

chunk = Document(
    page_content=('수신이 확인되기 전까지는 그 전자문서는 송신되지 않은 것으로 봅니다. 회사는 전자문\n'
 '서가 수신되지 않은 것을 확인한 경우에는 서면(등기우편 등)으로 다시 알려드립니다.- ⑤ 손해가 제1항 제1호 또는 제2호에 해당되는 '
 '사실로 생긴 것이 아님을 계약자 또는 피\n'
 '- 보험자가 증명한 경우에는 제4항에 관계없이 보상합니다.\n'
 '- ⑥ 회사는 다른 보험가입내역에 대한 계약 전․후 알릴 의무 위반을 이유로 계약을 해지하\n'
 '- 거나 보험금 지급을 거절하지 않습니다.'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000159',
              'chunk_char_len': 252,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
