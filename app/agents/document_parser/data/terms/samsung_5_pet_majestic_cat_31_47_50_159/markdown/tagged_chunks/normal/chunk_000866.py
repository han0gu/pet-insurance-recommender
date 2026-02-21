from langchain_core.documents import Document

chunk = Document(
    page_content=('여부가 판단된 경우, 이후 한국표준질병·사인분류 개정으로 상병분류가 변경되더라도 이 약\n'
 '관에서 보장하는 상병 해당 여부를 다시 판단하지 않습니다.- 152 -[별표-상해관련4] 아나필락시스 분류표약 관에 규정하는 '
 '아나필락시스로 분류되는 상병은 제9차 개정 한국표준질병·사인분류\n'
 '(통계청 고시 제2025-299호, 2026. 1. 1 시행) 중 다음에 적은 상병을 말하며, 이후 한국\n'
 '표준질병·사인분류가 개정되는 경우에는 개정된 기준에 따라 이 약관에서 보장하는 상\n'
 '병의 해당 여부를 판단합니다.| 분류항목 | 분류번호 |'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000866',
              'chunk_char_len': 291,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
