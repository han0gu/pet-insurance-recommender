from langchain_core.documents import Document

chunk = Document(
    page_content=('제3조(보험료의 납입)\n'
 '① 계약자는 새로이 증가된 보험의 목적에 대하여 일단위로 계산된 추가보험료를 납입하여 야 합니다. ② 새로이 증가된 보험의 목적의 '
 '보험기간이 시작된 후라도 다른 약정이 없으면 추가 보험 료를 받기 전에 생긴 손해는 보상하여 드리지 않습니다.\n'
 '제4조(준용규정)\n'
 '이 추가특별약관에 정하지 않은 사항은 보통약관 및 단체계약 특별약관을 따릅니다.'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 41},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000226',
              'chunk_char_len': 202,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
