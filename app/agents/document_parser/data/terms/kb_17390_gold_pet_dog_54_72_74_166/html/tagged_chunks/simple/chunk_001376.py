from langchain_core.documents import Document

chunk = Document(
    page_content=('제4항에 따라 변경된 약관을 적용하게 되어 보장내용이 변경되는 경우, 회사는 제<br>3항에도 불구하고 그 변경내용, 자동갱신 의사를 '
 '확인하는 내용 등을 갱신 전 보<br>장특약의 보험기간이 끝나기 15일 전까지 계약자에게 서면, 전화(음성녹음), 전자<br>문서, '
 '휴대전화 문자메시지 또는 이에 준하는 전자적 의사표시 등으로 2회 이상<br>안내하여 드립니다.<br>\uf000 회사는 제5항의 '
 '계약자의 자동갱신 의사를 전화(음성녹음), 직접 방문 또는 전자<br>적 의사표시(통신판매계약의 경우 통신수단)를 통해 확인하고, '
 '자동갱신'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_001376',
              'chunk_char_len': 294,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
