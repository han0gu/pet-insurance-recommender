from langchain_core.documents import Document

chunk = Document(
    page_content=('때</td><td>5</td></tr><tr><td>9) 한눈의 눈꺼풀에 뚜렷한 결손을 남긴 '
 '때</td><td>10</td></tr><tr><td>10) 한눈의 눈꺼풀에 뚜렷한 운동장해를 남긴 '
 "때</td><td>5</td></tr></tbody></table><footer id='5' "
 "style='font-size:14px'>177</footer><p id='6' data-category='paragraph' "
 "style='font-size:20px'>나"),
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
 'indexing': {'chunk_id': 'chunk_000920',
              'chunk_char_len': 255,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
