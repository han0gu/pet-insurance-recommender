from langchain_core.documents import Document

chunk = Document(
    page_content=('- 기재된 자기부담금을 차감한 후 보상비율을 곱한 금액이며 보험증권에 기재된 1일당\n'
 '- 보상한도액을 한도로 합니다. (자기부담금은 1일당 의료비에서 차감합니다.)\n'
 '<지급보험금의 계산>{(피보험자가 부담한 1일당 의료비 – 1일당 자기부담금) × 보상비율}과 보험증권에 기재된 1일당\n'
 '보상한도액 중 적은 금액<예시안내>[반려견 의료비(치과및구강질환포함)(수술당일제외, 검사비포함)(재가입형) 계산]- ∙ 보험가입금액 : '
 '10만원, 보상비율 : 70%, 자기부담금 : 3만원\n'
 '- ∙ 예시1'),
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
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['dental']},
 'indexing': {'chunk_id': 'chunk_000299',
              'chunk_char_len': 271,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
