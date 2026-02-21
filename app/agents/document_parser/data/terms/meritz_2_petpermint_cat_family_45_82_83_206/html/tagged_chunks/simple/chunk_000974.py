from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:20px'>2) 머리</h1><br><p id='69' data-category='paragraph' "
 "style='font-size:20px'>가) 손바닥 크기 1/2 이상의 반흔(흉터) 및 모발결손<br>나) 머리뼈의 손바닥 크기 1/2 "
 "이상의 손상 및 결손</p><br><h1 id='70' style='font-size:20px'>3) 목</h1><br><p "
 "id='71' data-category='paragraph' style='font-size:20px'>손바닥 크기 1/2 이상의 "
 '추상(추한'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['head']},
 'indexing': {'chunk_id': 'chunk_000974',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
