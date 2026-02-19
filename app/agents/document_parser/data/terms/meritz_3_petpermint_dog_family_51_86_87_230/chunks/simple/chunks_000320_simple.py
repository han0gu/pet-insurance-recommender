from langchain_core.documents import Document

chunk = Document(
    page_content=('구강질환 보장(구강질환의 치료 목적임에도 치아에 행해지는 치료는 보장하지 않습니 다)) ⑳ 아포퀠(Apoquel) 등의 JAK '
 'inhibitor(Janus kinase inhibitor) 약물'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 116},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['dental', 'digestive']},
 'indexing': {'chunk_id': 'chunk_000320',
              'chunk_char_len': 107,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
