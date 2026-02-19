from langchain_core.documents import Document

chunk = Document(
    page_content=('1회의 사고 | 1회의 사고라 함은 하나의 원인 또는 사실상 같은 종류의 위험에 계속적, 반복적 또는 누적 적으로 노출되어 그 결과로 '
 '발생한 사고로서 피보험자나 피해자의 수 또는 손해배상청구의 수에 관계없이 1회의 사고로 봅니다.'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 174},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000581',
              'chunk_char_len': 128,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
