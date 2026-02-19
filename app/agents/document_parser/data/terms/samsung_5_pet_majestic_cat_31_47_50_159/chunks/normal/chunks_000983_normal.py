from langchain_core.documents import Document

chunk = Document(
    page_content=('가) "뇌전증" 이라 함은 돌발적 뇌파이상을 나타내는 뇌질환으로 발작(경련, 의식장해 등)을 반복하는 것을 말한다. 나) 뇌전증 발작의 '
 '빈도 및 양상은 지속적인 항뇌전증제(항경련제) 약물로도 조 절되지 않는 뇌전증을 말하며, 진료기록에 기재되어 객관적으로 확인되는 뇌전증 '
 '발작의 빈도 및 양상을 기준으로 한다'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 148},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['head']},
 'indexing': {'chunk_id': 'chunk_000983',
              'chunk_char_len': 172,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
