from langchain_core.documents import Document

chunk = Document(
    page_content=(". 장해의 분류</h1><br><table id='4' style='font-size:16px'><thead><tr><td>장해의 "
 '분류</td><td>지급률</td></tr></thead><tbody><tr><td>1) 두눈이 멀었을 '
 '때</td><td>100</td></tr><tr><td>2) 한눈이 멀었을 때</td><td>50</td></tr><tr><td>3) '
 '한눈의 교정시력이 0.02 이하로 된 때</td><td>35</td></tr><tr><td>4) 한 눈의 교정시력이 0.06 이하로 된'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'eye']},
 'indexing': {'chunk_id': 'chunk_000918',
              'chunk_char_len': 281,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
