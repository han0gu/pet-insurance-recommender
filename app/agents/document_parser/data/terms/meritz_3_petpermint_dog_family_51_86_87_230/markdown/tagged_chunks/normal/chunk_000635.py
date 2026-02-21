from langchain_core.documents import Document

chunk = Document(
    page_content=('- (가벼운 정도)의 척추전만증(척추가 앞으로 휘어\n'
 '- 지는 증상), 척추후만증(척추가 뒤로 휘어지는\n'
 '- 증상) 또는 척추측만증(척추가 옆으로 휘어지는\n'
 '- 증상) 변형이 있을 때\n'
 '- 나) 척추체(척추뼈 몸통) 한 개의 압박률이 20%이상\n'
 '- 인 경우 또는 한 운동단위 내에 두 개 이상 척추\n'
 '- 체(척추뼈 몸통)의 압박골절로 각 척추체(척추뼈\n'
 '- 몸통)의 압박률의 합이 40% 이상일 때\n'
 '- 12) “추간판탈출증으로 인한 심한 신경 장해”란 추간\n'
 '- 판탈출증으로 추간판을 2마디이상(또는 1마디 추간판'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'head']},
 'indexing': {'chunk_id': 'chunk_000635',
              'chunk_char_len': 280,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
