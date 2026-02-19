from langchain_core.documents import Document

chunk = Document(
    page_content=('제8조 (준용규정)\n'
 '이 특별약관에 정하지 않은 사항은 3-1. 반려견 의료비(치과및구강질환포함)(수술당일제 외, 검사비포함)(재가입형) 특별약관을 따르며, '
 '3-1. 반려견 의료비(치과및구강질환포함)( 수술당일제외, 검사비포함)(재가입형) 특별약관에서 정하지 않은 사항은 특별약관 일반사 항을 '
 '따릅니다. 특별약관 일반사항에서도 정하지 않은 사항은 보통약관을 따릅니다.'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 67,
         'page': 86},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['dental']},
 'indexing': {'chunk_id': 'chunk_000551',
              'chunk_char_len': 204,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
