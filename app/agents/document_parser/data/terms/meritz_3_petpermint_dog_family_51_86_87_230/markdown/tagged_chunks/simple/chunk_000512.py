from langchain_core.documents import Document

chunk = Document(
    page_content=('안내하고자 할 경우에는 계약자에게 서면 또는 「전자서명\n'
 '법」 제2조 제2호에 따른 전자서명으로 동의를 얻어 수신확\n'
 '인을 조건으로 전자문서를 송신하여야 합니다. 계약자의 전\n'
 '자문서 수신이 확인되기 전까지는 그 전자문서는 송신되지\n'
 '않은 것으로 봅니다. 회사는 전자문서가 수신되지 않은 것\n'
 '을 확인한 경우에는 서면(등기우편 등)으로 다시 알려드립182니다.\n'
 '\uf000 손해가 제1항 제1호 또는 제2호에 해당되는 사실로 생긴\n'
 '것이 아님을 계약자 또는 피보험자가 증명한 경우에는 제4\n'
 '항에 관계없이 보상합니다.'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000512',
              'chunk_char_len': 277,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
