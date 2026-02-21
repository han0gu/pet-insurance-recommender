from langchain_core.documents import Document

chunk = Document(
    page_content=('원 또는 수족관에 상시고용된 수의사는 해당 농장, 동물원 또는 수족관의 동물\n'
 '에게 투여할 목적으로 처방대상 동물용 의약품에 대한 처방전을 발급할 수 있- 6 -다. 이 경우 상시고용된 수의사의 범위, 신고방법, '
 '처방전 발급 및 보존 방법,\n'
 '진료부 작성 및 보고, 교육, 준수사항 등 그 밖에 필요한 사항은 농림축산식품\n'
 '부령으로 정한다.제9조(보험금의 지급절차)① 회사는 제8조(보험금의 청구)에서 정한 서류를 접수한 때에는 접수증을 드리고 휴대전\n'
 '화 문자메시지 또는 전자우편 등으로도 송부하며, 그 서류를 접수한 날부터 3영업일 이'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000033',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
