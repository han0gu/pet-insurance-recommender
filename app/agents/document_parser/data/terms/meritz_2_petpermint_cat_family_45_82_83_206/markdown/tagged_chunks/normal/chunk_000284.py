from langchain_core.documents import Document

chunk = Document(
    page_content=('- 위한 비용\n'
 '- ⑤ 반려동물의 임신·출산, 제왕절개, 인공유산, 발정과\n'
 '- 관련된 비용 및 출산 후 증상 치료 비용\n'
 '- ⑥ 중성화, 불임 및 피임을 목적으로 한 수술 및 처치에\n'
 '- 따른 비용\n'
 '- ⑦ 미용으로 인한 비용\n'
 '- ⑧ 귀 성형, 꼬리 성형, 성대제거 및 미용성형을 위한\n'
 '- 수술 및 처치에 따른 비용\n'
 '- ⑨ 손톱절제(며느리발톱 제거 포함), 잔존유치, 잠복고\n'
 '- 환, 제대허니아(배꼽부위탈장), 항문낭 제거 등 건\n'
 '- 강동물에 실시하는 외과수술 및 기타 검사 또는 점\n'
 '- 안, 귀청소 등의 관리 비용'),
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
 'indexing': {'chunk_id': 'chunk_000284',
              'chunk_char_len': 284,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
