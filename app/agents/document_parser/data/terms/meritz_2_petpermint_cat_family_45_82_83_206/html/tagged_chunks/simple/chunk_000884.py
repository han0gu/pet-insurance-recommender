from langchain_core.documents import Document

chunk = Document(
    page_content=('KDA004</td><td>위 확장 및 염전 담즙성 구토 '
 '증후군</td></tr><tr><td>KDA005</td><td>유문협착증</td></tr><tr><td>KDA006</td><td>위장관 '
 '천공</td></tr><tr><td>KDA007</td><td>세균성 '
 '장염</td></tr><tr><td>KDA008</td><td>소장내 세균 과다 '
 '증식(SIBO)</td></tr><tr><td>KDA009</td><td>식이성 장 '
 '질환</td></tr><tr><td>KDA010</td><td>염증성 장'),
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
 'indexing': {'chunk_id': 'chunk_000884',
              'chunk_char_len': 281,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
