from langchain_core.documents import Document

chunk = Document(
    page_content=('| 24. 기타 골격 부분의 출산손상 | P13.8 |\n'
 '분류항목 분류번호\n'
 '25. 상세불명의 골격의 출산손상 P13.9주 ) 1. 제10차 개정 이후 이 약관에서 보장하는 상병의 해당 여부는 피보험자가 진단된 '
 '당시 시행되\n'
 '고 있는 한국표준질병·사인분류에 따라 판단합니다.\n'
 '2 . 진단 당시의 한국표준질병·사인분류에 따라 이 약관에서 보장하는 상병에 대한 보험금 지급\n'
 '여부가 판단된 경우, 이후 한국표준질병·사인분류 개정으로 상병분류가 변경되더라도 이 약'),
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
 'indexing': {'chunk_id': 'chunk_000862',
              'chunk_char_len': 252,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
