from langchain_core.documents import Document

chunk = Document(
    page_content=('- 8. 귀 성형, 꼬리 성형, 성대제거 및 미용성형을 위한 수술 및 처치에 따른 비용\n'
 '- 9. 손톱절제(며느리발톱 제거 포함), 유치잔존, 잠복고환, 제대허니아(배꼽부위\n'
 '- 탈장), 항문낭 제거 등 건강동물에 실시하는 외과수술 및 기타 검사 또는 점\n'
 '- 안, 귀청소 등의 관리 비용\n'
 '- 10. 입원중의 식이(食餌)에 해당하지 않는 음식물 및 식이요법, 그리고 수의사가\n'
 '- 처방하는 의약품 이외의 것(건강보조식품, 의약품지정이 되어 있지 않은 한방\n'
 '- 약, 의약부외품 등)'),
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
 'indexing': {'chunk_id': 'chunk_000563',
              'chunk_char_len': 264,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
