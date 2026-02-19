from langchain_core.documents import Document

chunk = Document(
    page_content=('\uf000 제3항에도 불구하고 회사가 계약자의 자동갱신 의사를 확인하지 못한 경우(계약자와 연락두절 등으로 회사 안내가 계약자에게 '
 '도달하지 못한 경우 포함)에는 갱신일 현재의 약관 등으로 갱신됩니다. 다만, 계약자는 갱신일 현재의 약 관 등에 대해 90일 이내에 그 '
 '계약을 취소할 수 있습니다.'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 190},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000650',
              'chunk_char_len': 160,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
