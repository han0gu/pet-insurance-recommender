from langchain_core.documents import Document

chunk = Document(
    page_content=('- (척추뼈 몸통)의 압박률은 인접 상ㆍ하부[인접 상ㆍ\n'
 '- 하부 척추체(척추뼈 몸통)에 진구성 골절이 있거나,\n'
 '- 다발성 척추골절이 있는 경우에는 골절된 척추와 가\n'
 '- 장 인접한 상ㆍ하부] 정상 척추체(척추뼈 몸통)의\n'
 '- 전방 높이의 평균에 대한 골절된 척추체(척추뼈 몸\n'
 '- 통) 전방 높이의 감소비를 압박률로 정한다.\n'
 '- 다) 척추(등뼈)의 기형장해는 ｢산업재해보상보험법 시행\n'
 '- 규칙｣상 경추부, 흉추부, 요추부로 구분하여 각각\n'
 '- 을 하나의 운동단위로 보며, 하나의 운동단위 내에'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000627',
              'chunk_char_len': 271,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
