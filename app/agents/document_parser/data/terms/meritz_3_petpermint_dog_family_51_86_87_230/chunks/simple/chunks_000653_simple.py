from langchain_core.documents import Document

chunk = Document(
    page_content=('. \uf000 다만, 해제 전에 발생한 보험금 지급사유에 대하여 회사 는 보상합니다. 이 경우 계약자는 즉시 갱신보장 보험료를 '
 '납입하여야 합니다. 만약 이 보험료를 납입하지 않으면 회 사는 지급할 보험금에서 이를 공제할 수 있습니다.'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 190},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000653',
              'chunk_char_len': 126,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
