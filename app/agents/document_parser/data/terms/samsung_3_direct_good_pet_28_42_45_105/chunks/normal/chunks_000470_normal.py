from langchain_core.documents import Document

chunk = Document(
    page_content=('5. 손톱의 절제(며느리발톱의 제거 포함), 잔존유치, 잠복고환, 배꼽허니아(배꼽부위탈장), 항문낭 제거 등 건강동물에 실시하는 외과수술 '
 '및 기타 검사 또는 손톱깎기 등의 처치비용 6. 미용으로 인한 비용 7. 귀 성형, 꼬리 성형, 성대 제거 및 미용성형 등 질병치료가 '
 '아닌 수술에 소요되는 비용 8. 건강식품, 보조식품, 보조치료제 및 Supplement 비용(치료를 목적으로 하는지 불 문합니다) 9. '
 '목욕 비용(약용 및 처방샴푸 값 포함) 및 벼룩, 진드기, 모낭충의 제거 비용 10'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 67,
         'page': 78},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000470',
              'chunk_char_len': 274,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
