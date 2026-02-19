from langchain_core.documents import Document

chunk = Document(
    page_content=('복)」에 따라 이 특별약관의 부활(효력회복)을 취급합니다.\n'
 '제 4조 (준용규정)\n'
 '이 특별약관에 정하지 않은 사항에 대하여는 보험계약을 따릅니다.\n'
 '【 붙임1】 특정신체부위 분류표\n'
 '구분 | 특 정 신 체 부 위\n'
 '1 | 위 .십이지장\n'
 '2 | 공장(빈창자), 회장(돌창자), 맹장(충수돌기 포함)\n'
 '3 | 대장(맹장, 직장 제외)\n'
 '4 | 직장\n'
 '5 | 항문\n'
 '6 | 간\n'
 '7 | 담낭(쓸개) 및 담관\n'
 '8 | 췌장\n'
 '9 | 비장\n'
 '10 | 기관, 기관지, 폐, 흉막 및 흉곽(늑골 포함)'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 129},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000837',
              'chunk_char_len': 261,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
