from langchain_core.documents import Document

chunk = Document(
    page_content=("id='59' style='font-size:20px'>2) 머리</h1><br><p id='60' "
 "data-category='paragraph' style='font-size:20px'>가) 손바닥 크기 이상의 반흔(흉터) 및 "
 "모발결손</p><footer id='61' style='font-size:14px'>184</footer><h1 id='62' "
 "style='font-size:20px'>나) 머리뼈의 손바닥 크기 이상의 손상 및 결손</h1><br><h1 id='63' "
 "style='font-size:20px'>3)"),
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
 'indexing': {'chunk_id': 'chunk_000971',
              'chunk_char_len': 294,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
