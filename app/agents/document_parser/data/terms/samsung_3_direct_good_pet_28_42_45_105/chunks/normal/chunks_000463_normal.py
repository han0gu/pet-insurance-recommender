from langchain_core.documents import Document

chunk = Document(
    page_content=('∙ 예시2\n'
 '- 피보험자가 이물질제거(구토유도약물) 시행 당일 부담한 의료비 : 33만원 - 반려견 의료비(치과및구강질환포함)(수술당일제외, '
 '검사비포함))(재가입형) 특별약관 지급금액 :\n'
 '10만원\n'
 '- 보험금 지급금액\n'
 '= [(33만원 - 3만원 - 10만원) × 70%, 20만원] 중 적은 금액 = 14만원\n'
 '∙ 예시3\n'
 '- 피보험자가 이물질제거(구토유도약물) 시행 당일 부담한 의료비 : 10만원 - 반려견 의료비(치과및구강질환포함)(수술당일제외, '
 '검사비포함))(재가입형) 특별약관 지급금액 :\n'
 '4.9만원\n'
 '- 보험금 미지급'),
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
 'clause': {'clause_type': 'exclusion',
            'risk_domains': ['dental', 'digestive']},
 'indexing': {'chunk_id': 'chunk_000463',
              'chunk_char_len': 290,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
