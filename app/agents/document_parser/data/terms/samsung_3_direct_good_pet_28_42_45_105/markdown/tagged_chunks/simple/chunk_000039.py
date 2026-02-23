from langchain_core.documents import Document

chunk = Document(
    page_content=('② 회사는 제1항의 통지로 인하여 위험의 변동이 발생한 경우에는 제21조(계약내용의 변\n'
 '경 등)에 따라 계약내용을 변경할 수 있습니다.<유의사항># [위험변경에 따른 계약변경 절차]위험변경사항 통지(우편, 전화, 방문 등)\n'
 '↓\n'
 '계약자, 피보험자의 계약변경사항 확인 후 청약\n'
 '↓\n'
 '계약변경사항 인수 심사\n'
 '↓\n'
 '정산금액 처리(환급 또는 추가납입)\n'
 '↓\n'
 '계약변경 완료![image](/image/placeholder)\n'
 '③ 회사는 제2항에 따라 계약내용을 변경할 때 위험이 감소된 경우에는 보험료를 감액하'),
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
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000039',
              'chunk_char_len': 275,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
