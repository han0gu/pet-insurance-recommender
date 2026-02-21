from langchain_core.documents import Document

chunk = Document(
    page_content=('- 기검진, 예방적 검사를 위한 비용\n'
 '- 5. 대상 반려동물의 정상적인 교배, 임신·출산, 제왕절개, 인공유산과 관련된 비\n'
 '- 용 및 출산 후 증상 치료 비용\n'
 '- 6. 중성화, 불임 및 피임을 목적으로 한 수술 및 처치에 따른 비용\n'
 '- 7. 미용으로 인한 비용\n'
 '- 8. 귀 성형, 꼬리 성형, 성대제거 및 미용성형을 위한 수술 및 처치에 따른 비용\n'
 '- 9. 손톱절제(며느리발톱 제거 포함), 유치잔존, 잠복고환, 제대허니아(배꼽부위\n'
 '- 탈장), 항문낭 제거 등 건강동물에 실시하는 외과수술 및 기타 검사 또는 점'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000605',
              'chunk_char_len': 286,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
