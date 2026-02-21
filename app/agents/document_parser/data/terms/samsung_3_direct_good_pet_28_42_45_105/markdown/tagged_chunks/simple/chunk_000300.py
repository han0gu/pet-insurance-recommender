from langchain_core.documents import Document

chunk = Document(
    page_content=('- ∙ 예시1\n'
 '- 피보험자가 부담한 1일당 의료비 13만원 (수술미발생, 의료비 중 검사비 5만원)\n'
 '- 보험금 지급금액\n'
 '= [(13만원 - 3만원) × 70%, 10만원] 중 적은 금액- \n'
 '= 7만원∙ 예시2- 67 -67 / 181- - 피보험자가 부담한 1일당 의료비 33만원 (수술미발생, 의료비 중 검사비 5만원)\n'
 '- - 보험금 지급금액\n'
 '= [(33만원 - 3만원) × 70%, 10만원] 중 적은 금액\n'
 '= 10만원- ⑥ 제1항의 「연간」이란 계약일로부터 매1년 단위로 도래하는 계약해당일 전일까지의\n'
 '- 기간을 의미합니다.'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000300',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
