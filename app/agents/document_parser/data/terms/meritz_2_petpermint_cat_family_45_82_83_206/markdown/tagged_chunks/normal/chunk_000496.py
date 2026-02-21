from langchain_core.documents import Document

chunk = Document(
    page_content=('| 6 | 소화기 질환 | KDA002 | 위 / 십이지장 궤양 |\n'
 '| 6 | 소화기 질환 | KDA003 KDA004 | 위 확장 및 염전 담즙성 구토 증후군 |\n'
 '| 6 | 소화기 질환 | KDA005 | 유문협착증 |\n'
 '| 6 | 소화기 질환 | KDA006 | 위장관 천공 |\n'
 '| 6 | 소화기 질환 | KDA007 | 세균성 장염 |\n'
 '| 6 | 소화기 질환 | KDA008 | 소장내 세균 과다 증식(SIBO) |\n'
 '| 6 | 소화기 질환 | KDA009 | 식이성 장 질환 |'),
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
 'indexing': {'chunk_id': 'chunk_000496',
              'chunk_char_len': 269,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
