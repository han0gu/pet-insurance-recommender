from langchain_core.documents import Document

chunk = Document(
    page_content=('제7조 (특별약관의 소멸)\n'
 '피보험자 또는 보험증권에 기재된 반려견이 보험기간 중에 사망하였을 경우에는 "보험료 및 해약환급금 산출방법서"에서 정하는 바에 따라 '
 '회사가 적립한 사망당시 이 특별약관의 계약자적립액 및 미경과보험료를 계약자에게 지급하고, 이 특별약관은 더 이상 효력이 없 습니다.\n'
 '제8조 (특별약관의 자동갱신)\n'
 '이 특별약관은 제도성 특별약관 4-1. [갱신형] 특별약관의 자동갱신 특별약관에 따라 갱 신됩니다.\n'
 '제9조 (준용규정)'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 94},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000611',
              'chunk_char_len': 246,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
