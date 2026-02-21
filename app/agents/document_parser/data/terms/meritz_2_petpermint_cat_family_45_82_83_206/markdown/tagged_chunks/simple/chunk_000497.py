from langchain_core.documents import Document

chunk = Document(
    page_content=('| 6 | 소화기 질환 | KDA009 | 식이성 장 질환 |\n'
 '| 6 | 소화기 질환 | KDA010 | 염증성 장 질환(IBD) |\n'
 '| 6 | 소화기 질환 | KDA011 | 단백 소실성 장증(PLE) |\n'
 '| 6 | 소화기 질환 | KDA012 | 장폐색 |\n'
 '| 6 | 소화기 질환 | KDA013 | 변비 (거대결장증 포함) |\n'
 '| 6 | 소화기 질환 | KDA014 | 모구증 (헤어볼 질환) |\n'
 '| 6 | 소화기 질환 | KDA015 | 장중첩 |\n'
 '172| 구 분 | 특정질병 | 분류코드 | 항목명 |'),
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
 'indexing': {'chunk_id': 'chunk_000497',
              'chunk_char_len': 285,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
