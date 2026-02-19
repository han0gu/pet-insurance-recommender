from langchain_core.documents import Document

chunk = Document(
    page_content=(': 파보 바이러스 감염, 디스템퍼 바이러스 감염, 파라 인플루엔자 감염, 전염성 간염, 아데노 바이러스 2 형 감염, 광견병, 코로나 '
 '바이러스 감염, 렙토스피 라 감염, 필라리아(심장사상충) 감염, 인플루엔자 감염'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 125},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000363',
              'chunk_char_len': 119,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.92}},
)
