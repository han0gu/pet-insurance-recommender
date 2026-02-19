from langchain_core.documents import Document

chunk = Document(
    page_content=('\uf000 이미 이 보장에서 후유장해보험금 지급사유에 해당되지 않았거나(보장개시 이전의 원인에 의하거나 또는 그 이전에 발생한 '
 '후유장해를 포함합니다), 후유장해보험금이 지급되 지 않았던 피보험자에게 그 신체의 동일 부위에 또다시 제6 항에 규정하는 후유장해상태가 '
 '발생하였을 경우에는 직전까 지의 후유장해에 대한 후유장해보험금이 지급된 것으로 보 고 최종 후유장해 상태에 해당되는 후유장해보험금에서 이 '
 '를 차감하여 지급합니다.\n'
 '제5조(보험금을 지급하지 않는 사유)'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 55},
 'term_type': 'basic',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000018',
              'chunk_char_len': 254,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
