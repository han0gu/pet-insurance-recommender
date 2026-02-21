from langchain_core.documents import Document

chunk = Document(
    page_content=('발정과<br>관련된 비용 및 출산 후 증상 치료 비용<br>⑥ 중성화, 불임 및 피임을 목적으로 한 수술 및 처치에<br>따른 '
 '비용<br>⑦ 미용으로 인한 비용<br>⑧ 귀 성형, 꼬리 성형, 성대제거 및 미용성형을 위한<br>수술 및 처치에 따른 비용<br>⑨ '
 '손톱절제(며느리발톱 제거 포함), 잔존유치, 잠복고<br>환, 제대허니아(배꼽부위탈장), 항문낭 제거 등 건</p><footer '
 "id='52' style='font-size:14px'>162</footer><p id='53' "
 "data-category='paragraph'"),
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
 'indexing': {'chunk_id': 'chunk_000812',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
