from langchain_core.documents import Document

chunk = Document(
    page_content=('[별표-상해관련4] 아나필락시스 분류표\n'
 '약 관에 규정하는 아나필락시스로 분류되는 상병은 제9차 개정 한국표준질병·사인분류 (통계청 고시 제2025-299호, 2026. 1. 1 '
 '시행) 중 다음에 적은 상병을 말하며, 이후 한국 표준질병·사인분류가 개정되는 경우에는 개정된 기준에 따라 이 약관에서 보장하는 상 병의 '
 '해당 여부를 판단합니다.\n'
 '분류항목 | 분류번호\n'
 '1 . 음식의 유해작용으로 인한 아나필락시스쇼크 | T78.0\n'
 '2 . 상세불명의 아나필락시스쇼크 | T78.2\n'
 '3 . 혈청에 의한 아나필락시스쇼크 | T80.5'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 153},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_001004',
              'chunk_char_len': 291,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
