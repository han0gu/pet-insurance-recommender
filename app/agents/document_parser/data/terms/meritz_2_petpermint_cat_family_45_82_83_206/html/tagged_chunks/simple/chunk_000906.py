from langchain_core.documents import Document

chunk = Document(
    page_content=(". 장해의 정의</h1><p id='36' data-category='list' style='font-size:16px'>1) “장해”라 "
 '함은 상해 또는 질병에 대하여 치유된 후<br>신체에 남아있는 영구적인 정신 또는 육체의 훼손상태<br>및 기능상실 상태를 말한다'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000906',
              'chunk_char_len': 149,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
