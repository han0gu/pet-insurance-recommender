from langchain_core.documents import Document

chunk = Document(
    page_content=('- 3. 중요한 사항: 계약 전 알릴 의무와 관련하여 회사가 그 사실을 알았더라면 계약의\n'
 '- 청약을 거절하거나 보험가입금액 한도 제한, 일부 보장 제외, 보험금 삭감, 보험료\n'
 '- 할증과 같이 조건부로 승낙하는 등 계약 승낙에 영향을 미칠 수 있는 사항을 말합\n'
 '- 니다.\n'
 '- 4. 한국표준질병∙사인분류 : 제9차 개정 한국표준질병·사인분류(통계청 고시 제2025-\n'
 '- 299호, 2026. 1. 1 시행)를 말하며 이후 한국표준질병·사인분류가 개정되는 경우는'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000153',
              'chunk_char_len': 255,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
