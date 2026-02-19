from langchain_core.documents import Document

chunk = Document(
    page_content=('제13조 (특별약관의 자동갱신)\n'
 '이 특별약관은 제도성 특별약관 5-1. [갱신형] 특별약관의 자동갱신 특별약관에 따라 갱 신됩니다.\n'
 '제14조 (준용규정)\n'
 '이 특별약관에 정하지 않은 사항은 4-1. 반려견 의료비(치과및구강질환포 함)(수술당일제 외, 검사비포함)(재가입형) 특별약관을 따르며, '
 '4-1. 반려견 의료비(치과및구강질환포함)( 수술당일제외, 검사비포함)(재가입형) 특별약관에서 정하지 않은 사항은 특별약관 일반사 항을 '
 '따릅니다. 특별약관 일반사항에서도 정하지 않은 사항은 보통약관을 따릅니다. 다만'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 123},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['dental']},
 'indexing': {'chunk_id': 'chunk_000775',
              'chunk_char_len': 283,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
