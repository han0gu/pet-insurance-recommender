from langchain_core.documents import Document

chunk = Document(
    page_content=('KCA003 | 거대 식도증 / 식도 확장증\n'
 'KDA001 | 위염 / 위장염 / 장염\n'
 'KDA002 | 위 / 십이지장 궤양\n'
 'KDA003 KDA004 | 위 확장 및 염전 담즙성 구토 증후군\n'
 'KDA005 | 유문협착증\n'
 'KDA006 | 위장관 천공\n'
 'KDA007 | 세균성 장염\n'
 'KDA008 | 소장내 세균 과다 증식(SIBO)\n'
 'KDA009 | 식이성 장 질환\n'
 'KDA010 | 염증성 장 질환(IBD)\n'
 'KDA011 | 단백 소실성 장증(PLE)\n'
 'KDA012 | 장폐색\n'
 'KDA013 | 변비 (거대결장증 포함)'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 172},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000606',
              'chunk_char_len': 281,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
