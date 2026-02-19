from langchain_core.documents import Document

chunk = Document(
    page_content=('7) “관절 하나의 기능을 완전히 잃었을 때”라 함은 아 래의 경우 중 하나에 해당하는 경우를 말한다.\n'
 '가) 완전 강직(관절굳음) 나) 근전도 검사상 완전손상(complete injury) 소 견이 있으면서 도수근력검사(MMT)에서 근력이'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 191},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['joint']},
 'indexing': {'chunk_id': 'chunk_000691',
              'chunk_char_len': 133,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
