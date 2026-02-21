from langchain_core.documents import Document

chunk = Document(
    page_content=('- ③ 제1항에서 정한「반려견의료비(치과및구강질환포함)(수술당일제외,검사비포함)보험금\n'
 '- 의 1일 한도」란 반려견이 이물제거(내시경) 또는 이물제거(구토유도약물)를 받은 당\n'
 '- 일 3-1. 반려견 의료비(치과및구강질환포함)(수술당일제외, 검사비포함)(재가입형) 특\n'
 '- 별약관에서 보상하는 의료비보험금의 1일 한도를 말합니다\n'
 '- ④ 제1항에서 정한「자기부담금」이란 보험증권에 기재된 3-1. 반려견 의료비(치과및구\n'
 '- 강질환포함)(수술당일제외, 검사비포함)(재가입형) 특별약관의 자기부담금을 말합니다'),
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
 'clause': {'clause_type': 'exclusion',
            'risk_domains': ['dental', 'digestive']},
 'indexing': {'chunk_id': 'chunk_000393',
              'chunk_char_len': 278,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
