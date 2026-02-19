from langchain_core.documents import Document

chunk = Document(
    page_content=('. 회사 가 전자문서로 안내하고자 할 경우에는 계약자에게 서면 또 는 「전자서명법」 제2조 제2호에 따른 전자서명으로 동의 를 얻어 '
 '수신확인을 조건으로 전자문서를 송신하여야 합니 다. 계약자의 전자문서 수신이 확인되기 전까지는 그 전자 문서는 송신되지 않은 것으로 '
 '봅니다. 회사는 전자문서가 수신되지 않은 것을 확인한 경우에는 서면(등기우편 등)으 로 다시 알려드립니다'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 62},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000071',
              'chunk_char_len': 206,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
