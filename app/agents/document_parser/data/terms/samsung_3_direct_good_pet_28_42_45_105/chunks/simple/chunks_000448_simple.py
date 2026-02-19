from langchain_core.documents import Document

chunk = Document(
    page_content=('④ 계약자는 제3항에 따른 재가입안내와 재가입여부 확인 요청을 받은 경우 재가입 의사 를 표시하여야 합니다. ⑤ 제3항 및 제4항에도 '
 '불구하고, 회사가 계약자의 재가입 의사를 확인하지 못한 경우(계 약자와의 연락두절로 회사의 안내가 계약자에게 도달하지 못한 경우 '
 '포함)에는 직전 계약과 동일한 조건으로 보험계약을 연장합니다'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 76},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000448',
              'chunk_char_len': 180,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
