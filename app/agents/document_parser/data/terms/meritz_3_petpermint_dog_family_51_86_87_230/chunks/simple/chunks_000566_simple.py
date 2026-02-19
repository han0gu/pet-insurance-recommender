from langchain_core.documents import Document

chunk = Document(
    page_content='된 비용 및 출산 후 증상 치료 비용',
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 171},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage',
            'risk_domains': ['other', 'urinary', 'digestive']},
 'indexing': {'chunk_id': 'chunk_000566',
              'chunk_char_len': 20,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
