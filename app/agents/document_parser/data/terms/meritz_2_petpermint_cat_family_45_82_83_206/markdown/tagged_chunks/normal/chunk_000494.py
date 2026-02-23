from langchain_core.documents import Document

chunk = Document(
    page_content=('| QCA001 | 귀 가려움증 (원인 불명) |  |  |\n'
 '| QFA001 | 발진 (원인 불명) |  |  |\n'
 '| QFA002 | 피부염 (원인 불명) |  |  |\n'
 '| QFA003 | 피부의 가려움증 (원인 불명) |  |  |\n'
 '| QFA004 | 탈모 (원인 불명) |  |  |\n'
 '| 6 | 소화기 질환 | ABB002 | 소화관 림프종 |\n'
 '| 6 | 소화기 질환 | ABA003 | 기타 소화기 계통의 양성 신생물 |\n'
 '| 6 | 소화기 질환 | ABB003 | 기타 소화기 계통의 악성 신생물 |'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'skin']},
 'indexing': {'chunk_id': 'chunk_000494',
              'chunk_char_len': 282,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
