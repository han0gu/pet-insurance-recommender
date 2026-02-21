from langchain_core.documents import Document

chunk = Document(
    page_content=('제47조 제1항 및 제3항의 정<br>상인의 신체 각 관절에 대한 평균 운동가능영역을 기준<br>으로 정상각도 및 측정방법 등을 '
 '따른다.</p><br><figure id=\'11\'><img style=\'font-size:20px\' alt="< 발가락 >" '
 'data-coord="top-left:(277,1051); bottom-right:(957,1576)" /></figure><footer '
 "id='12' style='font-size:14px'>198</footer><p id='13' "
 "data-category='paragraph'"),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['joint']},
 'indexing': {'chunk_id': 'chunk_001068',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
