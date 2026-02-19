from langchain_core.documents import Document

chunk = Document(
    page_content=('10) 뚜렷한 기형이란 다음 중 어느 하나에 해당하는 경 우를 말한다.\n'
 '가) 척추(등뼈)의 골절 또는 탈구 등으로 15° 이상 의 척추전만증(척추가 앞으로 휘어지는 증상), 척추후만증(척추가 뒤로 휘어지는 '
 '증상) 또는 10°이상의 척추측만증(척추가 옆으로 휘어지는 증상) 변형이 있을 때 나) 척추체(척추뼈 몸통) 한 개의 압박률이 40%이상 '
 '인 경우 또는 한 운동단위 내에 두 개 이상 척추'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 187},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['joint', 'other']},
 'indexing': {'chunk_id': 'chunk_000678',
              'chunk_char_len': 219,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.9}},
)
