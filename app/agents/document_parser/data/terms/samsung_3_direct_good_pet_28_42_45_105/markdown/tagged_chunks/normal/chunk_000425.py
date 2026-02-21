from langchain_core.documents import Document

chunk = Document(
    page_content=('특별약관의 부활(효력회복))를 따릅니다. 이 경우 부활(효력회복)일을 보험계약일로 하여\n'
 '제1조(보험금의 지급사유) 제3항을 적용합니다.# 제 9조 (준용규정)이 특별약관에 정하지 않은 사항은 3-1. 반려견 '
 '의료비(치과및구강질환포함)(수술당일제\n'
 '외, 검사비포함)(재가입형) 특별약관을 따르며, 3-1. 반려견 의료비(치과및구강질환포함)(\n'
 '수술당일제외, 검사비포함)(재가입형) 특별약관에서 정하지 않은 사항은 특별약관 일반사\n'
 '항을 따릅니다. 특별약관 일반사항에서도 정하지 않은 사항은 보통약관을 따릅니다. 다만'),
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
 'indexing': {'chunk_id': 'chunk_000425',
              'chunk_char_len': 284,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
