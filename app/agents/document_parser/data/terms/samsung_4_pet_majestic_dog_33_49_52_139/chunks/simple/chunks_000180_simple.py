from langchain_core.documents import Document

chunk = Document(
    page_content=('. 2. 장해: [별표2]장해분류표에서 정한 기준에 따른 장해상태를 말합니다. 3. 중요한 사항: 계약 전 알릴 의무와 관련하여 회사가 '
 '그 사실을 알았더라면 계약의 청약을 거절하거나 보험가입금액 한도 제한, 일부 보장 제외, 보험금 삭감, 보험료 할증과 같이 조건부로 '
 '승낙하는 등 계약 승낙에 영향을 미칠 수 있는 사항을 말합 니다. 4. 한국표준질병∙사인분류 : 제9차 개정 한국표준질병·사인분류(통계청 '
 '고시 제2025- 299호, 2026. 1'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 52},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000180',
              'chunk_char_len': 250,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
