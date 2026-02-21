from langchain_core.documents import Document

chunk = Document(
    page_content=('뚜렷한 결손을 남긴 때”라 함은 눈꺼풀<br>의 결손으로 눈을 감았을 때 각막(검은 자위)이 완전<br>히 덮이지 않는 경우를 '
 "말한다.</p><footer id='12' style='font-size:14px'>178</footer><p id='13' "
 "data-category='list' style='font-size:16px'>10) “눈꺼풀에 뚜렷한 운동장해를 남긴 때“라 함은 "
 '눈<br>을 떴을 때 동공을 1/2 이상 덮거나 또는 눈을 감았<br>을 때 각막을 완전히 덮을 수 없는 경우를 말한다.<br>11) '
 '외상이나 화상'),
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
 'indexing': {'chunk_id': 'chunk_000927',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
