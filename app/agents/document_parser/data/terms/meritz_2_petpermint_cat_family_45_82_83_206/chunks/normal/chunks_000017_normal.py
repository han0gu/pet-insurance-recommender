from langchain_core.documents import Document

chunk = Document(
    page_content=('우에는 그 때마다 이에 해당하는 후유장해지급률을 결정합 니다. 그러나 그 후유장해가 이미 후유장해보험금을 지급받 은 동일한 부위에 가중된 '
 '때에는 최종 장해상태에 해당하는 후유장해보험금에서 이미 지급받은 후유장해보험금을 차감 하여 지급합니다. 다만,【별표2(장해분류표)】의 각 '
 '신체부 위별 판정기준에서 별도로 정한 경우에는 그 기준에 따릅니 다.'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 51},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage',
            'risk_domains': ['head', 'joint', 'other']},
 'indexing': {'chunk_id': 'chunk_000017',
              'chunk_char_len': 192,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
