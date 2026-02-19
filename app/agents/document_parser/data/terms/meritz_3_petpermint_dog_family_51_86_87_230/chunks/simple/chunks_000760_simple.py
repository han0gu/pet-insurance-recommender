from langchain_core.documents import Document

chunk = Document(
    page_content=('하나에 해당하는 때를 말한다.\n'
 '가) 천장관절 또는 치골문합부가 분리된 상태로 치유 되었거나 좌골이 2.5cm 이상 분리된 부정유합 상태 나) 육안으로 변형(결손을 '
 '포함)을 명백하게 알 수 있을 정도로 방사선 검사로 측정한 각(角) 변형 이 20° 이상인 경우 다) 미골의 기형은 골절이나 탈구로 '
 '방사선 검사로 측 정한 각(角) 변형이 70° 이상 남은 상태'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 214},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['joint', 'other']},
 'indexing': {'chunk_id': 'chunk_000760',
              'chunk_char_len': 199,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
