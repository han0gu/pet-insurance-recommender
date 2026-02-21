from langchain_core.documents import Document

chunk = Document(
    page_content=('부 가 설 명 위험변경에 따른 계약 변경 절차\n'
 '위험변경사항 통지(우편, 전화, 방문 등)\n'
 '↓\n'
 '계약자, 피보험자의 계약변경사항 확인 후 청약\n'
 '↓\n'
 '계약변경사항 인수 심사\n'
 '↓\n'
 '정산금액 처리(환급 또는 추가납입)\n'
 '↓계약변경 완료\uf000 회사는 제2항에 따라 계약내용을 변경할 때 위험이 감소된 경우에는 보험료를 감\n'
 '액하고, 이후 기간 보장을 위한 재원인 계약자적립액 등의 차이로 인하여 발생한\n'
 '정산금액(이하 "정산금액"이라 합니다)을 환급하여 드립니다. 한편 위험이 증가'),
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
 'indexing': {'chunk_id': 'chunk_000065',
              'chunk_char_len': 256,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
