from langchain_core.documents import Document

chunk = Document(
    page_content=('- = 200만원\n'
 '⑥ 제1항의 「연간」이란 계약일로부터 매1년 단위로 도래하는 계약해당일 전일까지의기간을 의미합니다.⑦ 제2항에도 불구하고 3-1. 반려견 '
 '의료비(치과및구강질환포함)(수술당일제외, 검사비포- 81 -81 / 181함)(재가입형) 특별약관 제27조 (특별약관의 재가입에 관한 '
 '사항) 제1항 및 제2항에\n'
 '따라 재가입하는 경우 또는 3-1. 반려견 의료비(치과및구강질환포함)(수술당일제외,\n'
 '검사비포함)(재가입형) 특별약관 제27조 (특별약관의 재가입에 관한 사항) 제5항에 따'),
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
 'indexing': {'chunk_id': 'chunk_000436',
              'chunk_char_len': 272,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
