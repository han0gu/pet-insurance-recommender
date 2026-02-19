from langchain_core.documents import Document

chunk = Document(
    page_content=('수술 및 처치에 따른 비용 ⑦ 미용으로 인한 비용 ⑧ 귀 성형, 꼬리 성형, 성대제거 및 미용성형을 위한 수술 및 처치에 따른 비용 ⑨ '
 '손톱절제(며느리발톱 제거 포함), 잔존유치, 잠복고 환, 제대허니아(배꼽부위탈장), 항문낭 제거 등 건'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 162},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['head', 'other']},
 'indexing': {'chunk_id': 'chunk_000562',
              'chunk_char_len': 133,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
