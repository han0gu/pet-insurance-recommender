from langchain_core.documents import Document

chunk = Document(
    page_content=('1) 시력장해의 경우 공인된 시력검사표에 따라 최소 3회 이상 측정한다. 2) “교정시력”이라 함은 안경(콘택트렌즈를 포함한 모 든 '
 '종류의 시력 교정수단)으로 교정한 원거리 최대교 정시력을 말한다. 다만, 각막이식술을 받은 환자인 경 우 각막이식술 이전의 시력상태를 '
 '기준으로 평가한다. 3) “한 눈이 멀었을 때”라 함은 안구의 적출은 물론 명암을 가리지 못하거나(“광각무”) 겨우 가릴 수 있는 '
 '경우(“광각유”)를 말한다'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 203},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['eye']},
 'indexing': {'chunk_id': 'chunk_000704',
              'chunk_char_len': 236,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
