from langchain_core.documents import Document

chunk = Document(
    page_content=('. 척추체<br>(척추뼈 몸통)의 압박률은 인접 상ㆍ하부[인접 상ㆍ<br>하부 척추체(척추뼈 몸통)에 진구성 골절이 있거나,<br>다발성 '
 '척추골절이 있는 경우에는 골절된 척추와 가<br>장 인접한 상ㆍ하부] 정상 척추체(척추뼈 몸통)의<br>전방 높이의 평균에 대한 골절된 '
 '척추체(척추뼈 몸<br>통) 전방 높이의 감소비를 압박률로 정한다.<br>다) 척추(등뼈)의 기형장해는 ｢산업재해보상보험법 '
 '시행<br>규칙｣상 경추부, 흉추부, 요추부로 구분하여 각각<br>을 하나의 운동단위로 보며, 하나의 운동단위 내에<br>서 여러 개의'),
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
 'indexing': {'chunk_id': 'chunk_000986',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
