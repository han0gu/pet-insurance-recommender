from langchain_core.documents import Document

chunk = Document(
    page_content=('172| 구 분 | 특정질병 | 분류코드 | 항목명 |\n'
 '| --- | --- | --- | --- |\n'
 '|  |  | KDA016 | 소화관 기능 저하 (소화관 정체 포함) |\n'
 '| KDA017 | 항문낭염 / 항문낭 파열 |  |  |\n'
 '| KDA018 | 항문 주위 피부염 / 항문 주위 누공 |  |  |\n'
 '| KEA001 | 식도 탈장 |  |  |\n'
 '| KEA003 | 배꼽 탈장 |  |  |\n'
 '| KEA004 | 사타구니 탈장 (서혜부 탈장 포함) |  |  |\n'
 '| KEA005 | 회음부 탈장 |  |  |'),
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
 'indexing': {'chunk_id': 'chunk_000498',
              'chunk_char_len': 285,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
