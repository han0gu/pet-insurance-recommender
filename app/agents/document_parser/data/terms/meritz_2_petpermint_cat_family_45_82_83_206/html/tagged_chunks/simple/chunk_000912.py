from langchain_core.documents import Document

chunk = Document(
    page_content=(". 기타</h1><p id='40' data-category='paragraph' style='font-size:16px'>1) 하나의 "
 '장해가 관찰방법에 따라서 장해분류표상 2가지<br>이상의 신체부위에서 장해로 평가되는 경우에는 그 중<br>높은 지급률을 '
 "적용한다.</p><footer id='41' style='font-size:14px'>176</footer><p id='0' "
 "data-category='list' style='font-size:16px'>2) 동일한 신체부위에 2가지 이상의 장해가 발생한 "
 '경우에<br>는 합산하지'),
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
 'indexing': {'chunk_id': 'chunk_000912',
              'chunk_char_len': 300,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
