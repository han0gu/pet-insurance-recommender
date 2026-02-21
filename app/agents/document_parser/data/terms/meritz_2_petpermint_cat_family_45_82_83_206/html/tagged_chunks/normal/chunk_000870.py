from langchain_core.documents import Document

chunk = Document(
    page_content=('(원인 불명)</td></tr><tr><td>QGA002</td><td>요실금 (원인 불명)</td></tr><tr><td>QGA003 '
 'QGA004</td><td>비정상 성분의 소변 (원인 불명) 핍뇨 (원인 불명)</td></tr><tr><td '
 'rowspan="23">5</td><td '
 'rowspan="23"></td><td>AFA001</td><td>지방종</td></tr><tr><td>AFA002</td><td>조직구종 '
 '(피부)</td></tr><tr><td>AFA003</td><td>유두종'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['skin']},
 'indexing': {'chunk_id': 'chunk_000870',
              'chunk_char_len': 279,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
