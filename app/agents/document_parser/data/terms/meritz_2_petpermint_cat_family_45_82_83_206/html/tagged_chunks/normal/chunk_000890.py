from langchain_core.documents import Document

chunk = Document(
    page_content=('소화기계 감염증</td></tr><tr><td>KGA009</td><td>소화계통의 기타 질환</td></tr><tr><td>PAA014 '
 'PAA015</td><td>고양이 파보 바이러스(FPV) 고양이 코로나 바이러스 '
 '감염</td></tr><tr><td>QEA001</td><td>구토 (원인 '
 '불명)</td></tr><tr><td>QEA002</td><td>설사 / 혈변 (원인 '
 '불명)</td></tr><tr><td>QEA003</td><td>복통 (원인 '
 '불명)</td></tr><tr><td>QEA004</td><td>복수 (원인'),
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
 'indexing': {'chunk_id': 'chunk_000890',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
