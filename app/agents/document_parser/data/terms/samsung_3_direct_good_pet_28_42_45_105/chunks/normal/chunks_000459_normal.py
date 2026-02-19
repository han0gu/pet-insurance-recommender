from langchain_core.documents import Document

chunk = Document(
    page_content=('. 반려견 의료비(치과및구 강질환포함)(수술당일제외, 검사비포함)(재가입형) 특별약관의 자기부담금을 말합니다 ⑤ 회사가 지급할 제1항에서 '
 '정한 보험금은 피보험자가 부담한 당일 발생한 의료비에서 제4항에서 정한「자기부담금」및 제3항에서 정한「반려견의료비(치과및구강질환포함 '
 ')(수술당일제외,검사비포함)보험금의 1일 한도」를 차감한 후 보상비율을 곱한 금액으 로 제1항에서 정한 보상한도액을 한도로 합니다. 단, '
 '3-1'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 67,
         'page': 77},
 'term_type': 'special',
 'clause': {'clause_type': 'deductible', 'risk_domains': ['dental']},
 'indexing': {'chunk_id': 'chunk_000459',
              'chunk_char_len': 231,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
