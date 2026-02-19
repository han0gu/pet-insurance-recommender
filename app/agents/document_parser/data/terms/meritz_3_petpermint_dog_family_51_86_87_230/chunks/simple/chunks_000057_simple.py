from langchain_core.documents import Document

chunk = Document(
    page_content='등으로 주로 사용하는 경우에 한함)하게 된 경우(다만, 전동 휠체어, 의료용 스쿠터 등 보행보조용 의자차는 제외 합니다.)',
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 63},
 'term_type': 'basic',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000057',
              'chunk_char_len': 68,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
