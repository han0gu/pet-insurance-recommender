from langchain_core.documents import Document

chunk = Document(
    page_content=('. 8) 상기 장해항목에 해당되지 않는 장기간의 간병이 필요 한 만성질환(만성간질환, 만성폐쇄성폐질환 등)은 장 해의 평가 대상으로 '
 '인정하지 않는다.'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 225},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive', 'other']},
 'indexing': {'chunk_id': 'chunk_000806',
              'chunk_char_len': 83,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
