from langchain_core.documents import Document

chunk = Document(
    page_content=('∙ 예시2\n'
 '- 피보험자가 부담한 수술당일 의료비 303만원 - 보험금 지급금액\n'
 '= [(303만원 - 3만원) × 70%, 200만원] 중 적은 금액 = 200만원\n'
 '⑥ 제1항의 「연간」이란 계약일로부터 매1년 단위로 도래하는 계약해당일 전일까지의\n'
 '기간을 의미합니다.\n'
 '⑦ 제2항에도 불구하고 3-1. 반려견 의료비(치과및구강질환포함)(수술당일제외, 검사비포'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 81},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion',
            'risk_domains': ['dental', 'digestive']},
 'indexing': {'chunk_id': 'chunk_000504',
              'chunk_char_len': 197,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
