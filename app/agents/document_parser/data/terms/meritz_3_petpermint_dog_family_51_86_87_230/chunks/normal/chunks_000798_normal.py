from langchain_core.documents import Document

chunk = Document(
    page_content=('. 8) 발가락 관절의 운동범위 측정은 장해평가시점의 ｢산업 재해보상보험법 시행규칙｣ 제47조 제1항 및 제3항의 정 상인의 신체 각 '
 '관절에 대한 평균 운동가능영역을 기준 으로 정상각도 및 측정방법 등을 따른다.'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 223},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['joint', 'other']},
 'indexing': {'chunk_id': 'chunk_000798',
              'chunk_char_len': 118,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
