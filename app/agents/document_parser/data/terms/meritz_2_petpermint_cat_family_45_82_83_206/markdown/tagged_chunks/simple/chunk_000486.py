from langchain_core.documents import Document

chunk = Document(
    page_content=('| OAA011 | 피부질환 | 고양이 특발성 방광염(FIC) |  |\n'
 '| OAA012 | 피부질환 | 고양이 하부 비뇨기계 질환(FLUTD) |  |\n'
 '| OAA013 | 피부질환 | 고양이 하부 요로계 증후군(FUS) |  |\n'
 '| OAA014 | 피부질환 | 기타 비뇨기계 질환 |  |\n'
 '| OAA015 | 피부질환 | 다낭성 신장 질환 |  |\n'
 '| OAA016 | 피부질환 | 단백 소실성 신증(PLN) |  |\n'
 '| QGA001 | 피부질환 | 혈뇨 (원인 불명) |  |'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other',
            'risk_domains': ['digestive', 'skin', 'urinary']},
 'indexing': {'chunk_id': 'chunk_000486',
              'chunk_char_len': 266,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
