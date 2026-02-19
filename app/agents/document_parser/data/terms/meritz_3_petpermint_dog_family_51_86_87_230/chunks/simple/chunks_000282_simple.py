from langchain_core.documents import Document

chunk = Document(
    page_content=('항목 | 자기부담금 | 지급 한도\n'
 '통원 의료비 | 통원 중 수술을 하지 않은 날의 경우 | 1일당 3만원/5만원 중 보험증권에 기재된 자기부담금 | 1일당 30만원\n'
 '통원 중 수술을 한 날의 경우 | 수술당일에 한하여 1일당 200만원'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 108},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000282',
              'chunk_char_len': 130,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
