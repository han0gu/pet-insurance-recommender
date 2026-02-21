from langchain_core.documents import Document

chunk = Document(
    page_content=('| OAA003 | 피부질환 | 수신증 |  |\n'
 '| OAA004 | 피부질환 | 만성 신장 질환 (신부전 포함) |  |\n'
 '| OAA005 | 피부질환 | 신장 결석 |  |\n'
 '| OAA006 OAA007 | 피부질환 | 방광염 방광 결석 |  |\n'
 '| OAA008 | 피부질환 | 요도 폐색 |  |\n'
 '| OAA009 | 피부질환 | 요로 결석증 |  |\n'
 '| OAA010 | 피부질환 | 신경성 배뇨 이상 |  |\n'
 '| OAA011 | 피부질환 | 고양이 특발성 방광염(FIC) |  |'),
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
            'risk_domains': ['digestive', 'head', 'skin', 'urinary']},
 'indexing': {'chunk_id': 'chunk_000485',
              'chunk_char_len': 268,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
