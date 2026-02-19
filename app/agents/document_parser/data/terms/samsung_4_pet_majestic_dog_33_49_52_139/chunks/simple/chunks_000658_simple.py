from langchain_core.documents import Document

chunk = Document(
    page_content=('6. 미용으로 인한 비용 7. 귀 성형, 꼬리 성형, 성대 제거 및 미용성형 등 질병치료가 아닌 수술에 소요되는 비용 8. 건강식품, '
 '보조식품, 보조치료제 및 Supplement 비용(치료를 목적으로 하는지 불 문합니다) 9. 목욕 비용(약용 및 처방샴푸 값 포함) 및 '
 '벼룩, 진드기, 모낭충의 제거 비용 10. 한방 및 한약(보상하는 상해 또는 질병의 치료를 위한 침술 및 물리치료는 제외합 니다), '
 '온천요법, 산소요법, 면역요법 등의 대체적 처치에 의한 치료를 위한 비용 11'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 111},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000658',
              'chunk_char_len': 268,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
