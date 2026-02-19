from langchain_core.documents import Document

chunk = Document(
    page_content=('운동범위 제한 및 무릎관절(슬관절)의 동요성 등으로 평가한다.\n'
 '가) 각 관절의 운동범위 측정은 장해평가시점의 ｢산 업재해보상보험법 시행규칙｣ 제47조 제1항 및 제 3항의 정상인의 신체 각 관절에 '
 '대한 평균 운동 가능영역을 기준으로 정상각도 및 측정방법 등을 따른다. 나) 관절기능장해가 신경손상으로 인한 경우에는 운 동범위 측정이 '
 '아닌 근력 및 근전도 검사를 기준 으로 평가한다.\n'
 '7) “관절 하나의 기능을 완전히 잃었을 때”라 함은 아 래의 경우 중 하나에 해당하는 때를 말한다.'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 219},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage',
            'risk_domains': ['digestive', 'head', 'joint']},
 'indexing': {'chunk_id': 'chunk_000777',
              'chunk_char_len': 270,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
