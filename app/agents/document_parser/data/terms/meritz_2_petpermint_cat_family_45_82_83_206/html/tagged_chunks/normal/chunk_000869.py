from langchain_core.documents import Document

chunk = Document(
    page_content=('하부 비뇨기계 질환(FLUTD)</td></tr><tr><td>OAA013</td><td>고양이 하부 요로계 '
 '증후군(FUS)</td></tr><tr><td>OAA014</td><td>기타 비뇨기계 '
 '질환</td></tr><tr><td>OAA015</td><td>다낭성 신장 '
 '질환</td></tr><tr><td>OAA016</td><td>단백 소실성 '
 '신증(PLN)</td></tr><tr><td>QGA001</td><td>혈뇨 (원인 '
 '불명)</td></tr><tr><td>QGA002</td><td>요실금 (원인'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'urinary']},
 'indexing': {'chunk_id': 'chunk_000869',
              'chunk_char_len': 284,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
