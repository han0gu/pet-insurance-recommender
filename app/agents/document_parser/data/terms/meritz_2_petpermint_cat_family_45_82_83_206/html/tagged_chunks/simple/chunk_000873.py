from langchain_core.documents import Document

chunk = Document(
    page_content=('또는 악성이 불확실한)</td></tr><tr><td>AFB009</td><td>피부 '
 '림프종</td></tr><tr><td>AFB010</td><td>편평세포암종</td></tr><tr><td>AFA011</td><td>항문주위선종</td></tr><tr><td>AFB012</td><td>항문주위선암종</td></tr><tr><td>AFA013</td><td>상세미상의 '
 '피부 신생물 (양성)</td></tr><tr><td>AFB013</td><td>상세미상의 피부 신생물'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'skin']},
 'indexing': {'chunk_id': 'chunk_000873',
              'chunk_char_len': 264,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
