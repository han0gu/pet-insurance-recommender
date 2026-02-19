from langchain_core.documents import Document

chunk = Document(
    page_content=('. 회사가 전자문서로 안내하고자 할 경우에는 계약자에게 서면 또는 「전자서명 법」 제2조 제2호에 따른 전자서명으로 동의를 얻어 수신확 '
 '인을 조건으로 전자문서를 송신하여야 합니다. 계약자의 전 자문서 수신이 확인되기 전까지는 그 전자문서는 송신되지 않은 것으로 봅니다. '
 '회사는 전자문서가 수신되지 않은 것 을 확인한 경우에는 서면(등기우편 등)으로 다시 알려드립'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 182},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000618',
              'chunk_char_len': 202,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
