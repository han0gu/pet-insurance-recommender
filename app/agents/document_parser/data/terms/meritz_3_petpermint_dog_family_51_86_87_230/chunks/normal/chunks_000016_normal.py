from langchain_core.documents import Document

chunk = Document(
    page_content=('. \uf000 같은 상해로 두 가지 이상의 후유장해가 생긴 경우에는 후유장해 지급률을 합산하여 지급합니다. 다만,【별표2(장 '
 '해분류표)】의 각 신체부위별 판정기준에 별도로 정한 경우 에는 그 기준에 따릅니다. \uf000 다른 상해로 인하여 후유장해가 2회 이상 '
 '발생하였을 경'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 54},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage',
            'risk_domains': ['head', 'joint', 'other']},
 'indexing': {'chunk_id': 'chunk_000016',
              'chunk_char_len': 145,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
