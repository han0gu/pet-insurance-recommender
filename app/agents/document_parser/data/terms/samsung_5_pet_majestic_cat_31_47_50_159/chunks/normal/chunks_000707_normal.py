from langchain_core.documents import Document

chunk = Document(
    page_content=('④ 제3항의 「자기부담금」 이란 보험증권에 기재된 4-1. 반려묘 의료비(치과및구강질환 포함)(재가입형) 특별약관의 자기부담금을 말합니다 '
 '⑤ 제1항의 「연간」이란 계약일로부터 매1년 단위로 도래하는 계약해당일 전일까지의 기간을 의미하며 이물제거(내시경) 보험금지급횟수와 '
 '이물제거(구토유도약물) 보험금 지급횟수를 합산하여 연간2회를 한도로 합니다. ⑥ 제2항에도 불구하고 4-1. 반려묘 '
 '의료비(치과및구강질환포함)(재가입형) 특별약관 제 27조 (특별약관의 재가입에 관한 사항) 제1항 및 제2항에 따라 재가입하는 경우 또는 '
 '4-1'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 113},
 'term_type': 'special',
 'clause': {'clause_type': 'deductible',
            'risk_domains': ['dental', 'digestive']},
 'indexing': {'chunk_id': 'chunk_000707',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
