from langchain_core.documents import Document

chunk = Document(
    page_content=('- 사. 미용으로 인한 비용\n'
 '- 아. 귀 성형, 꼬리 성형, 성대제거 및 미용성형을 위한 수술 및 처치에 따른 비용\n'
 '- 자. 손톱절제(며느리발톱 제거 포함), 잔존유치, 잠복고환, 배꼽허니아(배꼽부위탈장), 항문낭 제\n'
 '- 거 등 건강동물에 실시하는 외과수술 및 기타 검사 또는 점안, 귀청소 등의 관리 비용\n'
 '- 6 -당신에게 좋은보험 삼성화재차. 입원중의 식이(食餌)에 해당하지 않는 음식물 및 식이요법, 그리고 수의사가 처방하는 의약\n'
 '품 이외의 것(건강보조식품, 의약품지정이 되어 있지 않은 한방약, 의약부외품 등)'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000016',
              'chunk_char_len': 289,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
