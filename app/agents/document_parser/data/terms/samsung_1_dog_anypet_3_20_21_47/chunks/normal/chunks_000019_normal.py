from langchain_core.documents import Document

chunk = Document(
    page_content=('. 미용으로 인한 비용 6. 귀 성형, 꼬리 성형, 성대 제거 및 미용성형 등 질병치료가 아닌 수술에 소요되는 비용 7. 치석제거 및 '
 '치과치료비용(부정 교합 기타 이상형성의 개선치료 포함) 8. 건강식품, 보조식품, 보조치료제 및 Supplement 비용(치료를 목적으로 '
 '하는지 불문합니다.) 9. 목욕 비용(약용 및 처방샴푸 값 포함) 및 벼룩, 진드기, 모낭충의 제거 비용 10'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 6},
 'term_type': 'basic',
 'clause': {'clause_type': 'exclusion',
            'risk_domains': ['head', 'dental', 'skin', 'other']},
 'indexing': {'chunk_id': 'chunk_000019',
              'chunk_char_len': 213,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
