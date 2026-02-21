from langchain_core.documents import Document

chunk = Document(
    page_content=('- 이상전위가 있는 상태\n'
 '- 8) 약간의 운동장해\n'
 '- 머리뼈(두개골)와 상위목뼈(상위경추: 제1, 2경추)를\n'
 '- 제외한 척추체(척추뼈 몸통)에 골절 또는 탈구로 2개\n'
 '- 의 척추체(척추뼈 몸통)를 유합(아물어 붙음) 또는 고\n'
 '- 정한 상태\n'
 '- 9) 심한 기형이란 다음 중 어느 하나에 해당하는 경우를\n'
 '- 말한다.\n'
 '- 가) 척추(등뼈)의 골절 또는 탈구 등으로 35° 이상\n'
 '- 의 척추전만증(척추가 앞으로 휘어지는 증상),\n'
 '- 척추후만증(척추가 뒤로 휘어지는 증상) 또는\n'
 '- 20° 이상의 척추측만증(척추가 옆으로 휘어지는'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion',
            'risk_domains': ['digestive', 'head', 'joint']},
 'indexing': {'chunk_id': 'chunk_000558',
              'chunk_char_len': 291,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
