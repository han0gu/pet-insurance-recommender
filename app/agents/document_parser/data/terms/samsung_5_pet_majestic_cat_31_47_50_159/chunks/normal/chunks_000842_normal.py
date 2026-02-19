from langchain_core.documents import Document

chunk = Document(
    page_content=('48 | 상 ·하악골(위·아래턱뼈)\n'
 '49 | 쇄골\n'
 '50 | 늑골(갈비뼈)\n'
 '【 붙임2】 특정질병 분류표\n'
 '약관에 규정하는 특정질병으로 분류되는 질병은 제9차 개정 한국표준질병·사인분류(통계 청 고시 제2025-299호, 2026. 1. 1 '
 '시행) 중 다음에 적은 질병을 말합니다.\n'
 '구분 | 대 상 질 병 | 분 류 번 호\n'
 '51 | 담석증 | K80 | 담석증\n'
 '52 | 요로결석증 | N20 | 신장 및 요관의 결석\n'
 'N21 | 하부요로의 결석\n'
 'N22 | 달리 분류된 질환에서의 요로의 결석\n'
 'N23 | 상세불명의 신장 급통증'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 130},
 'term_type': 'special',
 'clause': {'clause_type': 'definition',
            'risk_domains': ['other', 'other', 'digestive', 'urinary']},
 'indexing': {'chunk_id': 'chunk_000842',
              'chunk_char_len': 289,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
