from langchain_core.documents import Document

chunk = Document(
    page_content=('고양이칼리시바이러스 감염증, 고양이바이러스성비기관지염, 고양이백혈병 바이러스감염증, 고양이헤르페스바이러스감염증, 고 양이클라미디아감염증 '
 '③ 상병명을 알 수 없는 상해 또는 질병에 대한 치료 ④ 백신 접종비용 및 기타 질병예방을 위한 검사 또는 투약·예방 접종비용 및 '
 '정기검진, 예방적 검사를 위한 비용 ⑤ 반려동물의 임신·출산, 제왕절개, 인공유산, 발정과 관련된 비용 및 출산 후 증상 치료 비용 ⑥ '
 '중성화, 불임 및 피임을 목적으로 한 수술 및 처치에 따른 비용 ⑦ 미용으로 인한 비용 ⑧ 귀 성형, 꼬리 성형, 성대제거 및'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 162},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000561',
              'chunk_char_len': 295,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
