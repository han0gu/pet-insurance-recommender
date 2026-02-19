from langchain_core.documents import Document

chunk = Document(
    page_content=('QFA001 | 발진 (원인 불명)\n'
 'QFA002 | 피부염 (원인 불명)\n'
 'QFA003 | 피부의 가려움증 (원인 불명)\n'
 'QFA004 | 탈모 (원인 불명)\n'
 '6 | 소화기 질환 | ABB002 | 소화관 림프종\n'
 'ABA003 | 기타 소화기 계통의 양성 신생물\n'
 'ABB003 | 기타 소화기 계통의 악성 신생물\n'
 'ABC003 | 기타 소화기 계통의 신생물(양성 또는 악성이 불확실한)\n'
 'KCA001 | 식도염\n'
 'KCA002 | 식도 협착 / 식도 폐색\n'
 'KCA003 | 거대 식도증 / 식도 확장증\n'
 'KDA001 | 위염 / 위장염 / 장염'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 172},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['skin', 'digestive']},
 'indexing': {'chunk_id': 'chunk_000605',
              'chunk_char_len': 294,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
