from langchain_core.documents import Document

chunk = Document(
    page_content=('연간 두번째 이상 | 1일당 15만원\n'
 'MRI,CT 및 내시경처치를 받지 않은 날의 경우 | 1일당 15만원\n'
 '통원 중 수술을 한 날의 경우 | 수술당일에 한하여 1일당 250만원'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 151},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000485',
              'chunk_char_len': 98,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
