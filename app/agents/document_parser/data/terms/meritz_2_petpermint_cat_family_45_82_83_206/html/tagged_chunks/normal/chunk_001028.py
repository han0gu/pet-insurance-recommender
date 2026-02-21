from langchain_core.documents import Document

chunk = Document(
    page_content=(". 장해의 분류</h1><br><table id='50' style='font-size:16px'><thead><tr><td>장해의 "
 '분류</td><td>지급률</td></tr></thead><tbody><tr><td>1) 두다리의 발목이상을 잃었을 '
 '때</td><td>100</td></tr><tr><td>2) 한다리의 발목이상을 잃었을 '
 '때</td><td>60</td></tr><tr><td>3) 한다리의 3대관절중 관절 하나의 기능을 완전히 잃었 을 '
 '때</td><td>30</td></tr><tr><td>4) 한다리의 3대관절중 관절'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'joint']},
 'indexing': {'chunk_id': 'chunk_001028',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
