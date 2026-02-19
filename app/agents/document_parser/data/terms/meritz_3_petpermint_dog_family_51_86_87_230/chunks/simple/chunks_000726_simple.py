from langchain_core.documents import Document

chunk = Document(
    page_content='4) “씹어먹는 기능에 약간의 장해를 남긴 때“라 함은 아래의 경우 중 하나 이상에 해당되는 때를 말한다.',
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 207},
 'term_type': 'special',
 'clause': {'clause_type': 'other',
            'risk_domains': ['dental', 'joint', 'other']},
 'indexing': {'chunk_id': 'chunk_000726',
              'chunk_char_len': 59,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.9}},
)
