from langchain_core.documents import Document

chunk = Document(
    page_content=('11) 약간의 기형이란 다음 중 어느 하나에 해당하는 경 우를 말한다.\n'
 '가) 1개 이상의 척추(등뼈)의 골절 또는 탈구로 경도 (가벼운 정도)의 척추전만증(척추가 앞으로 휘어 지는 증상), 척추후만증(척추가 '
 '뒤로 휘어지는 증상) 또는 척추측만증(척추가 옆으로 휘어지는 증상) 변형이 있을 때 나) 척추체(척추뼈 몸통) 한 개의 압박률이 '
 '20%이상 인 경우 또는 한 운동단위 내에 두 개 이상 척추 체(척추뼈 몸통)의 압박골절로 각 척추체(척추뼈 몸통)의 압박률의 합이 '
 '40% 이상일 때'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 213},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['joint', 'other']},
 'indexing': {'chunk_id': 'chunk_000756',
              'chunk_char_len': 271,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
