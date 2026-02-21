from langchain_core.documents import Document

chunk = Document(
    page_content=('험료의 환급)에 따른 보험료를 계약자에게 지급합니다. 회사가 전자문서로 안내하고자\n'
 '할 경우에는 계약자에게 서면 또는 「전자서명법」 제2조 제2호에 따른 전자서명으로\n'
 '동의를 얻어 수신확인을 조건으로 전자문서를 송신하여야 합니다. 계약자의 전자문서\n'
 '수신이 확인되기 전까지는 그 전자문서는 송신되지 않은 것으로 봅니다. 회사는 전자문\n'
 '서가 수신되지 않은 것을 확인한 경우에는 서면(등기우편 등)으로 다시 알려드립니다.⑤ 손해가 제1항 제1호 또는 제2호에 해당되는 사실로 '
 '생긴 것이 아님을 계약자 또는 피'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000154',
              'chunk_char_len': 279,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
