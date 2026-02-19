from langchain_core.documents import Document

chunk = Document(
    page_content=('. 3) “치유된 후”라 함은 상해 또는 질병에 대한 치료의 효과를 기대할 수 없게 되고 또한 그 증상이 고정된 상태를 말한다. 4) '
 '다만, 영구히 고정된 증상은 아니지만 치료종결후 한시 적으로 나타나는 장해에 대하여는 그 기간이 5년 이상 인 경우 해당장해 지급률의 '
 '20%를 장해지급률로 한다'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 201},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000695',
              'chunk_char_len': 164,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
