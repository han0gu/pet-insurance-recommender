from langchain_core.documents import Document

chunk = Document(
    page_content=('시 제2025-299호, 2026. 1. 1 시행) 중 다음에 적은 상병을 말하며, 이후 한국표준질병\n'
 '· 사인분류가 개정되는 경우는 개정된 기준에 따라 이 약관에서 보장하는 상병의 해당\n'
 '여부를 판단합니다.| 분류항목 | 분류번호 |\n'
 '| --- | --- |\n'
 '| 1. 두개골 및 안면골의 골절 | S02 |\n'
 '| 2. 머리의 으깸손상 | S07 |\n'
 '| 3. 머리의 상세불명 손상 | S09.9 |\n'
 '| 4. 목의 골절 | S12 |\n'
 '| 5. 늑골, 흉골 및 흉추의 골절 | S22 |\n'
 '| 6. 요추 및 골반의 골절 | S32 |'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive', 'head']},
 'indexing': {'chunk_id': 'chunk_000854',
              'chunk_char_len': 292,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
