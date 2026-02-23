from langchain_core.documents import Document

chunk = Document(
    page_content=('- 변경 등)에 따라 계약내용을 변경할 수 있습니다.\n'
 '[위험변경에 따른 계약변경 절차]<유의사항># 위험변경사항 통지(우편, 전화, 방문 등)↓# 계약자, 피보험자의 계약변경사항 확인 후 '
 '청약![image](/image/placeholder)\n'
 '↓계약변경사항 인수 심사↓정산금액 처리(환급 또는 추가납입)↓계약변경 완료- ③ 회사는 제2항에 따라 계약내용을 변경할 때 위험이 감소된 '
 '경우에는 보험료를 감액하\n'
 '- 고, 이후 기간 보장을 위한 재원인 계약자적립액 등의 차이로 인하여 발생한 정산금액'),
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
 'indexing': {'chunk_id': 'chunk_000056',
              'chunk_char_len': 275,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
