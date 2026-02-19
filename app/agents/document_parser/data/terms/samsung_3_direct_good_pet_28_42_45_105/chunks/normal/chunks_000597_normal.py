from langchain_core.documents import Document

chunk = Document(
    page_content=('제6조 (특별약관의 자동갱신)\n'
 '이 특별약관은 제도성 특별약관 4-1. [갱신형] 특별약관의 자동갱신 특별약관에 따라 갱 신됩니다.\n'
 '제7조 (준용규정)\n'
 '이 특별약관에 정하지 않은 사항은 특별약관 일반사항을 따릅니다. 특별약관 일반사항에 서도 정하지 않은 사항은 보통약관을 따릅니다.'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 67,
         'page': 92},
 'term_type': 'special',
 'clause': {'clause_type': 'renewal', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000597',
              'chunk_char_len': 156,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
