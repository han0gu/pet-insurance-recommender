from langchain_core.documents import Document

chunk = Document(
    page_content=('를 따릅니다. 이 경우 부활(효력회복)일을 보험계약일로 하여 제1조(보험금의 지급사유)\n'
 '제3항을 적용합니다.# 제7조 (특별약관의 자동갱신)이 특별약관은 제도성 특별약관 5-1. [갱신형] 특별약관의 자동갱신 특별약관에 따라 '
 '갱\n'
 '신됩니다.# 제8조 (준용규정)이 특별약관에 정하지 않은 사항은 4-1. 반려묘 의료비(치과및 구강질환포함)(재가입형)\n'
 '특별약관을 따르며, 4-1. 반려묘 의료비(치과및구강질환포함)(재가입형) 특별약관에서 정\n'
 '하지 않은 사항은 특별약관 일반사항을 따릅니다. 특별약관 일반사항에서도 정하지 않은'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'renewal', 'risk_domains': ['dental']},
 'indexing': {'chunk_id': 'chunk_000587',
              'chunk_char_len': 291,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
