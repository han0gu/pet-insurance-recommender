from langchain_core.documents import Document

chunk = Document(
    page_content=('치아, 기존 의치(틀니, 임플란트 등)의 결손<br>은 치아의 상실로 인정하지 않는다.<br>14) 상실된 치아의 크기가 크든지 또는 '
 '치간의 간격이나<br>치아 배열구조 등의 문제로 사고와 관계없이 새로<br>운 치아가 결손된 경우에는 사고로 결손된 치아 수<br>에 '
 "따라 지급률을 결정한다.<br>15) 어린이의 유치는 향후에 영구치로 대체되므로 후유</p><footer id='48' "
 "style='font-size:14px'>183</footer><h1 id='49' style='font-size:20px'>장해의 "
 '대상이 되지'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['dental', 'digestive']},
 'indexing': {'chunk_id': 'chunk_000963',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
