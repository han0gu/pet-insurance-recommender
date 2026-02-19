from langchain_core.documents import Document

chunk = Document(
    page_content=('. 11) 뇌‧중추신경계 손상(정신‧인지기능 저하, 편마비 등) 으로 인한 말하는 기능의 장해(실어증, 구음장애) 또는 씹어먹는 기능의 '
 '장해는 신경계‧정신행동 장해 평가와 비교하여 그 중 높은 지급률 하나만 인정한 다. 12) “치아의 결손”이란 치아의 상실 또는 발치된 '
 '경우 를 말하며, 치아의 일부 손상으로 금관치료(크라운 보철수복)를 시행한 경우에는 치아의 일부 결손을 인정하여 1/2개 결손으로 '
 '적용한다'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 208},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['head', 'dental']},
 'indexing': {'chunk_id': 'chunk_000732',
              'chunk_char_len': 229,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
