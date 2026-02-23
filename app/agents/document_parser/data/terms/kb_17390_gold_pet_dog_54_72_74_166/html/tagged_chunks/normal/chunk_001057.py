from langchain_core.documents import Document

chunk = Document(
    page_content=('. 의료비 금액이 기재된 영수증(사업자등록된 업체가 발행한 영수증으로, 사업<br>자등록번호를 포함하여야 하며, 카드전표 또는 국세청에 '
 '통보된 현금영수증이<br>어야 합니다.)<br>6. 신분증(주민등록증이나 운전면허증 등 사진이 붙은 정부기관발행 신분증, '
 '본<br>상<br>인이 아닌 경우에는 본인의 인감증명서, 본인서명사실확인서 또는 안전성과 해<br>신뢰성이 확보된 전자적 수단을 활용한 '
 '보험수익자 의사표시의 확인방법 포 및<br>함) 질<br>7'),
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
 'indexing': {'chunk_id': 'chunk_001057',
              'chunk_char_len': 253,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
