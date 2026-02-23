from langchain_core.documents import Document

chunk = Document(
    page_content=('가입형) 특별약관 1일당 보상한도액보다 적을 경우 보험금을 지급하지 않습니다.# <예시안내>∙ 보상한도액 : 200만원(20만원), '
 '보상비율 70% 기준\n'
 '반려견 의료비(치과및구강질환포함)(수술당일제외, 검사비포함)(재가입형) 보상한도액 10만원, 자\n'
 '기부담금 3만원, 보상비율 70% 기준∙ 예시1- 피보험자가 이물질제거(내시경) 시행 당일 부담한 의료비 : 153만원\n'
 '- 반려견 의료비(치과및구강질환포함)(수술당일제외, 검사비포함)(재가입형) 특별약관 지급금액 :'),
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
 'clause': {'clause_type': 'limit', 'risk_domains': ['dental']},
 'indexing': {'chunk_id': 'chunk_000396',
              'chunk_char_len': 260,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
