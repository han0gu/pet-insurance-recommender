from langchain_core.documents import Document

chunk = Document(
    page_content=('완전 강직(관절굳음)<br>나) 근전도 검사상 완전손상(complete injury) 소<br>견이 있으면서 도수근력검사(MMT)에서 '
 "근력이</p><footer id='33' style='font-size:14px'>191</footer><p id='34' "
 "data-category='paragraph' style='font-size:20px'>“0등급(Zero)”인 경우</p><br><p "
 "id='35' data-category='paragraph' style='font-size:20px'>8) “관절 하나의 기능에 심한 "
 '장해를'),
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
 'indexing': {'chunk_id': 'chunk_001018',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
