from langchain_core.documents import Document

chunk = Document(
    page_content=('3) 치매\n'
 '가) “치매”라 함은 정상적으로 성숙한 뇌가 질병이 나 외상 후 기질성 손상으로 파괴되어 한번 획득 한 지적기능이 지속적 또는 전반적으로 '
 '저하되는 것을 말한다.'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 203},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['head']},
 'indexing': {'chunk_id': 'chunk_000745',
              'chunk_char_len': 95,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
