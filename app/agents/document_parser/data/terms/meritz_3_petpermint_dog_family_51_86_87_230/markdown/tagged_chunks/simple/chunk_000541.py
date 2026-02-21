from langchain_core.documents import Document

chunk = Document(
    page_content=('는 보상합니다. 이 경우 계약자는 즉시 갱신보장 보험료를\n'
 '납입하여야 합니다. 만약 이 보험료를 납입하지 않으면 회\n'
 '사는 지급할 보험금에서 이를 공제할 수 있습니다.제5조(자동갱신 적용대상 계약의 보장개시)제2조(자동갱신 적용대상 계약의 자동갱신)에 '
 '따라 계약이\n'
 '갱신되는 경우 갱신보장계약의 보장개시는 갱신일 당일부터\n'
 '개시됩니다.190제6조(준용규정)이 특별약관에서 정하지 않은 사항은 보통약관 및 해당 특'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000541',
              'chunk_char_len': 226,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
