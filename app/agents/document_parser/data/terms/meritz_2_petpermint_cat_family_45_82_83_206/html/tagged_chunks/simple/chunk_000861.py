from langchain_core.documents import Document

chunk = Document(
    page_content=('계통의 신생물(양성 또는 악성이 불확 '
 '실한)</td></tr><tr><td>HAA001</td><td>고혈압</td></tr><tr><td>HAA002 '
 'HAA003</td><td>저혈압 부정맥</td></tr><tr><td>HAA004</td><td>판막 질환(의증, 심잡음 '
 '포함)</td></tr><tr><td>HAA005</td><td>판막 질환 (심부전 '
 '증상)</td></tr><tr><td>HAA006</td><td>심비대 (원인 '
 '불명)</td></tr><tr><td>HAA007</td><td>확장성'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000861',
              'chunk_char_len': 284,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
