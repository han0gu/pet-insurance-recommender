from langchain_core.documents import Document

chunk = Document(
    page_content=('- 우 각막이식술 이전의 시력상태를 기준으로 평가한다.\n'
 '- 3) “한 눈이 멀었을 때”라 함은 안구의 적출은 물론\n'
 '- 명암을 가리지 못하거나(“광각무”) 겨우 가릴 수\n'
 '- 있는 경우(“광각유”)를 말한다.\n'
 '- 4) “한눈의 교정시력이 0.02이하로 된 때”라 함은 안\n'
 '- 전수동(Hand Movement)주1), 안전수지(Finger\n'
 '- Counting)주2) 상태를 포함한다.\n'
 '※ 주1) 안전수동 : 물체를 감별할 정도의 시력상태\n'
 '가 아니며 눈앞에서 손의 움직임을 식별할\n'
 '수 있을 정도의 시력상태'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['eye']},
 'indexing': {'chunk_id': 'chunk_000593',
              'chunk_char_len': 278,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
