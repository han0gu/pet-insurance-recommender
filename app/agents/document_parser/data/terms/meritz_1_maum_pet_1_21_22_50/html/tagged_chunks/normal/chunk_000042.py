from langchain_core.documents import Document

chunk = Document(
    page_content=('. 상병명을 알 수 없는 상해 또는 질병에 대한 치료<br>6. 백신 접종비용 및 기타 질병예방을 위한 검사 또는 투약·예방 접종비용 및 '
 '정기검<br>진, 예방적 검사를 위한 비용<br>7. 반려동물의 임신·출산, 인공유산, 발정과 관련된 비용 및 출산 후 증상 '
 '치료비용<br>8. 중성화, 불임 및 피임을 목적으로 한 처치에 따른 비용<br>9. 미용으로 인한 비용<br>10. 귀 성형, 꼬리 '
 '성형, 성대제거 및 미용성형을 위한 처치에 따른 비용<br>11'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000042',
              'chunk_char_len': 257,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
