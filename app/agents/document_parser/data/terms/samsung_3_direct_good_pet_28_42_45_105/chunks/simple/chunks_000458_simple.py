from langchain_core.documents import Document

chunk = Document(
    page_content=('② 이물제거(내시경)과 이물제거(구토유도약물)을 동일한 날에 받은 경우 이물제거(내시 경) 보험금만 지급됩니다. ③ 제1항에서 '
 '정한「반려견의료비(치과및구강질환포함)(수술당일제외,검사비포함)보험금 의 1일 한도」란 반려견이 이물제거(내시경) 또는 '
 '이물제거(구토유도약물)를 받은 당 일 3-1. 반려견 의료비(치과및구강질환포함)(수술당일제외, 검사비포함)(재가입형) 특 별약관에서 '
 '보상하는 의료비보험금의 1일 한도를 말합니다 ④ 제1항에서 정한「자기부담금」이란 보험증권에 기재된 3-1'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 77},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion',
            'risk_domains': ['dental', 'digestive']},
 'indexing': {'chunk_id': 'chunk_000458',
              'chunk_char_len': 269,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
