from langchain_core.documents import Document

chunk = Document(
    page_content=('때 (손가락 하나마다)</td><td>10</td></tr><tr><td>4) 한손의 5개손가락 모두의 손가락뼈 일부를 잃었을 때 또는 '
 '뚜렷한 장해를 남긴 때</td><td>30</td></tr><tr><td>5) 한손의 첫째 손가락의 손가락뼈 일부를 '
 "잃었을</td><td>10</td></tr></tbody></table><footer id='74' "
 "style='font-size:14px'>195</footer><table id='75' "
 "style='font-size:16px'><thead><tr><td>장해의"),
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
 'indexing': {'chunk_id': 'chunk_001049',
              'chunk_char_len': 288,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
