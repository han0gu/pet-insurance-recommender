from langchain_core.documents import Document

chunk = Document(
    page_content=('- 존 방법, 진료부 작성 및 보고, 교육, 준수사항 등 그 밖에 필요한 사항은 농림축산식품부령으로\n'
 '- 정한다.\n'
 '# 제9조 (보험금의 지급절차)- ① 회사는 제8조(보험금의 청구)에서 정한 서류를 접수한 때에는 접수증을 드리고 휴대전\n'
 '- 화 문자메시지 또는 전자우편 등으로 송부하며, 그 서류를 접수한 날부터 3영업일 이\n'
 '- 내에 보험금을 지급합니다.\n'
 '- ② 회사가 보험금 지급사유를 조사 ·확인하기 위해 필요한 기간이 제1항의 지급기일을\n'
 '- 초과할 것이 명백히 예상되는 경우에는 그 구체적 사유와 지급예정일 및 보험금 가지'),
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
 'indexing': {'chunk_id': 'chunk_000479',
              'chunk_char_len': 293,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
