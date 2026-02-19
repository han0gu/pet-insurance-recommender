from langchain_core.documents import Document

chunk = Document(
    page_content=('\uf000 제3조(보험금의 지급사유)에서 장해지급률이 상해 발생 일부터 180일 이내에 확정되지 않는 경우에는 상해 발생일 부터 '
 '180일이 되는 날의 의사 진단에 기초하여 고정될 것으 로 인정되는 상태를 장해지급률로 결정합니다. 다만,【별표 2(장해분류표)】에 '
 '장해판정시기를 별도로 정한 경우에는 그에 따릅니다'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 50},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000013',
              'chunk_char_len': 169,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
