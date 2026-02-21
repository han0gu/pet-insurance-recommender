from langchain_core.documents import Document

chunk = Document(
    page_content=('| 6 | 소화기 질환 | ABB003 | 기타 소화기 계통의 악성 신생물 |\n'
 '| 6 | 소화기 질환 | ABC003 | 기타 소화기 계통의 신생물(양성 또는 악성이 불확실한) |\n'
 '| 6 | 소화기 질환 | KCA001 | 식도염 |\n'
 '| 6 | 소화기 질환 | KCA002 | 식도 협착 / 식도 폐색 |\n'
 '| 6 | 소화기 질환 | KCA003 | 거대 식도증 / 식도 확장증 |\n'
 '| 6 | 소화기 질환 | KDA001 | 위염 / 위장염 / 장염 |\n'
 '| 6 | 소화기 질환 | KDA002 | 위 / 십이지장 궤양 |'),
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
 'indexing': {'chunk_id': 'chunk_000495',
              'chunk_char_len': 290,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
