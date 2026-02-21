from langchain_core.documents import Document

chunk = Document(
    page_content=("id='27' style='font-size:16px'><thead><tr><td>장해의 "
 '분류</td><td>지급률</td></tr></thead><tbody><tr><td>를 남긴 때 6) 한팔의 3대관절중 관절 하나의 '
 '기능에 약간의 장해 를 남긴 때 7) 한팔에 가관절이 남아 뚜렷한 장해를 남긴 때 8) 한팔에 가관절이 남아 약간의 장해를 남긴 때 9) '
 "한팔의 뼈에 기형을 남긴 때</td><td>5 20 10 5</td></tr></tbody></table><h1 id='28' "
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
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'joint']},
 'indexing': {'chunk_id': 'chunk_001012',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
