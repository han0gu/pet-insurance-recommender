from langchain_core.documents import Document

chunk = Document(
    page_content=('를 전화(음성녹음), 직접 방문 또는 전자적 의사표시, 통신판매계약의 경우 통신수단\n'
 '을 통해 확인합니다.- ④ 계약자는 제3항에 따른 재가입안내와 재가입여부 확인 요청을 받은 경우 재가입 의사\n'
 '- 를 표시하여야 합니다.\n'
 '- ⑤ 제3항 및 제4항에도 불구하고, 회사가 계약자의 재가입 의사를 확인하지 못한 경우(계\n'
 '- 약자와의 연락두절로 회사의 안내가 계약자에게 도달하지 못한 경우 포함)에는 직전\n'
 '- 계약과 동일한 조건으로 보험계약을 연장합니다. 다만, 보험계약이 연장된 경우 연장'),
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
 'clause': {'clause_type': 'renewal', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000385',
              'chunk_char_len': 269,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
