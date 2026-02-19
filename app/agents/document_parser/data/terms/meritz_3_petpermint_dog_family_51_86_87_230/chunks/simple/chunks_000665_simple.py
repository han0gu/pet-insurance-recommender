from langchain_core.documents import Document

chunk = Document(
    page_content=('. \uf000 제1항의 규정에도 불구하고 다음 중 어느 하나의 사유로 보험계약에서 정한 보험금의 지급사유가 발생한 경우 회사 는 '
 '보험금을 지급하여 드립니다.'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 193},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000665',
              'chunk_char_len': 83,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
