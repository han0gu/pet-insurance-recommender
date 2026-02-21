from langchain_core.documents import Document

chunk = Document(
    page_content=('- 을 지급합니다.\n'
 '- 4. 동물장묘업자가 제공하는 장례확인서(동물장묘업소 등록번호, 업소명 및 주\n'
 '- 소, 전화번호, 서비스 대상 동물의 종류, 품종, 나이, 장례서비스 이용일자,\n'
 '- 화장서비스 등의 서비스 이용내역, 비용 등 포함)\n'
 '- 5. 장례비용 영수증(사업자등록된 업체가 발행한 영수증으로, 사업자등록번호를\n'
 '- 포함하여야 하며, 카드전표 또는 국세청에 통보된 현금영수증이어야 합니다.)\n'
 '- 6. 신분증(주민등록증이나 운전면허증 등 사진이 붙은 정부기관 발행 신분증, 본'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000649',
              'chunk_char_len': 267,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
