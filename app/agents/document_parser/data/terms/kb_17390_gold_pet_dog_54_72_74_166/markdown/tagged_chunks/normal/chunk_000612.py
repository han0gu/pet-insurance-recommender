from langchain_core.documents import Document

chunk = Document(
    page_content=('- 된 치료비 세부내역 포함\n'
 '- - "항암약물치료" 시행한 경우 : 항암약물치료에 해당하는 약물이 명시된 수\n'
 '- 질\n'
 '- 의사처방전\n'
 '- 병\n'
 '- 5. 의료비 금액이 기재된 영수증(사업자등록된 업체가 발행한 영수증으로, 사업\n'
 '- 자등록번호를 포함하여야 하며, 카드전표 또는 국세청에 통보된 현금영수증이\n'
 '- 어야 합니다.)\n'
 '- 6. 신분증(주민등록증이나 운전면허증 등 사진이 붙은 정부기관발행 신분증, 본\n'
 '- 상\n'
 '- 인이 아닌 경우에는 본인의 인감증명서, 본인서명사실확인서 또는 안전성과 해'),
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
 'indexing': {'chunk_id': 'chunk_000612',
              'chunk_char_len': 270,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
