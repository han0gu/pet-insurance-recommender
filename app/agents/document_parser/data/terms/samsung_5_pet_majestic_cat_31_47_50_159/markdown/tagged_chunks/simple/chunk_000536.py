from langchain_core.documents import Document

chunk = Document(
    page_content=('- 가입 요건, 보장내용 변경내역, 보험료 수준, 재가입 절차 및 재가입 의사 여부를 확인\n'
 '- 하는 내용 등을 서면(등기우편 등), 전화(음성녹음), 전자문서, 휴대전화 문자메시지\n'
 '- 또는 이에 준하는 전자적 의사표시 등으로 알려드리고, 회사는 계약자의 재가입의사\n'
 '- 를 전화(음성녹음), 직접 방문 또는 전자적 의사표시, 통신판매계약의 경우 통신수단\n'
 '- 을 통해 확인합니다.\n'
 '- ④ 계약자는 제3항에 따른 재가입안내와 재가입여부 확인 요청을 받은 경우 재가입 의사\n'
 '- 를 표시하여야 합니다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000536',
              'chunk_char_len': 275,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
