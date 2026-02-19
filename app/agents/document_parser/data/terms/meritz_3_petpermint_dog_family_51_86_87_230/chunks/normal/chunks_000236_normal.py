from langchain_core.documents import Document

chunk = Document(
    page_content='\uf000 회사는 계약자가 제1회 보험료를 납입한 때부터 1년 이 상 지난 유효한 계약으로서 그 보험종목의 변경을 요청할',
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 100},
 'term_type': 'special',
 'clause': {'clause_type': 'renewal', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000236',
              'chunk_char_len': 63,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
