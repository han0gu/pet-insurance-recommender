from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다) 척추(등뼈)의 기형장해는 ｢산업재해보상보험법 시행 규칙｣상 경추부, 흉추부, 요추부로 구분하여 각각 을 하나의 운동단위로 보며, '
 '하나의 운동단위 내에 서 여러 개의 척추체(척추뼈 몸통)에 압박골절이 발 생한 경우에는 각 척추체(척추뼈 몸통)의 압박률을 합산하고, 두 '
 '개 이상의 운동단위에서 장해가 발생 한 경우에는 그 중 가장 높은 지급률을 적용한다.'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 211},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['joint']},
 'indexing': {'chunk_id': 'chunk_000745',
              'chunk_char_len': 201,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
