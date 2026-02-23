from langchain_core.documents import Document

chunk = Document(
    page_content=('소화기 계통의 신생물(양성 또는 악성이 '
 '불확실한)</td></tr><tr><td>KCA001</td><td>식도염</td></tr><tr><td>KCA002</td><td>식도 '
 '협착 / 식도 폐색</td></tr><tr><td>KCA003</td><td>거대 식도증 / 식도 '
 '확장증</td></tr><tr><td>KDA001</td><td>위염 / 위장염 / '
 '장염</td></tr><tr><td>KDA002</td><td>위 / 십이지장 궤양</td></tr><tr><td>KDA003 '
 'KDA004</td><td>위 확장 및 염전'),
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
 'indexing': {'chunk_id': 'chunk_000883',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
