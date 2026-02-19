from langchain_core.documents import Document

chunk = Document(
    page_content=('제9조 (준용규정)\n'
 '이 추가특별약관에 정하지 않은 사항은 4-1. 반려묘 의료비(치과및구강질환포함)(재가입 형) 특별약관을 따르며, 4-1. 반려묘 '
 '의료비(치과및구강질환포함)(재가입형) 특별약관에서 정하지 않은 사항은 특별약관 일반사항을 따릅니다. 특별약관 일반사항에서도 정하지 않 은 '
 '사항은 보통약관을 따릅니다. 다만, 보통약관 제10조(환급금의 중도인출), 제11조(만 기환급금의 지급)은 제외합니다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 120},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['dental']},
 'indexing': {'chunk_id': 'chunk_000779',
              'chunk_char_len': 226,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
