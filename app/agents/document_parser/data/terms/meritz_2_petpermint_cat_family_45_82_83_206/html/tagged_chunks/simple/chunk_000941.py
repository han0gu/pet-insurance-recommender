from langchain_core.documents import Document

chunk = Document(
    page_content=('눈을 감고 일어서기 곤란하거나 두 눈 을 뜨고 10m 거리를 직선으로 걷다가 '
 '쓰</td><td>20</td></tr><tr><td>경우</td><td>12</td></tr><tr><td>러지는 두 눈을 뜨고 '
 '10미터 거리를 직선으로 걷 다가 중간에 균형을 잡으려 멈추어야 하 는 경우 두 눈을 뜨고 10m 거리를 직선으로 걸을 때 중앙에서 '
 "60cm 이상 벗어나는 경우</td><td>8</td></tr></tbody></table><br><p id='27' "
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
 'clause': {'clause_type': 'other', 'risk_domains': ['eye']},
 'indexing': {'chunk_id': 'chunk_000941',
              'chunk_char_len': 281,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
