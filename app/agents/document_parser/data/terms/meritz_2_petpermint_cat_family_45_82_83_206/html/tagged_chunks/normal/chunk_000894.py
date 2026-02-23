from langchain_core.documents import Document

chunk = Document(
    page_content=('/ 근첨농양</td></tr><tr><td>JBA007</td><td>기타 구강 질환 '
 '치아흡수성병변(FORL)</td></tr><tr><td>JBA008</td><td>고양이 만성 '
 '구내염(FCGs)</td></tr><tr><td rowspan="2">8</td><td rowspan="2">전신성 '
 '질환</td><td></td><td></td></tr><tr><td>PAA018</td><td>고양이 '
 "전염성복막염(FIP)</td></tr></tbody></table><footer id='21'"),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['dental']},
 'indexing': {'chunk_id': 'chunk_000894',
              'chunk_char_len': 275,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
