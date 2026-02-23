from langchain_core.documents import Document

chunk = Document(
    page_content=('수준, 재가입 절차 및 재가입 의사<br>여부를 확인하는 내용 등을 서면(등기우편 등), 전화(음성녹음), 전자문서, 휴대<br>전화 '
 '문자메시지 또는 이에 준하는 전자적 의사표시 등으로 알려드리고, 회사는<br>계약자의 재가입의사를 전화(음성녹음), 직접 방문 또는 '
 '전자적 의사표시, 통신<br>판매계약의 경우 통신수단을 통해 확인합니다.<br>\uf000 계약자는 제3항에 따른 재가입안내와 재가입여부 '
 '확인 요청을 받은 경우 재가입<br>의사를 표시하여야 합니다.<br>\uf000 제3항 및 제4항에도 불구하고, 회사가 계약자의 재가입 '
 '의사를'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000923',
              'chunk_char_len': 295,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
