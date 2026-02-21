from langchain_core.documents import Document

chunk = Document(
    page_content=('- 을 지급합니다. 해\n'
 '- 및\n'
 '- 4. 사망을 확인할 수 있는 서류(동물폐사확인서, 동물화장증명서 등)\n'
 '- 질\n'
 '- 5. 신분증(주민등록증이나 운전면허증 등 사진이 붙은 정부기관 발행 신분증, 본\n'
 '- 병\n'
 '- 인이 아닌 경우에는 본인의 인감증명서, 본인서명사실확인서 또는 안전성과\n'
 '- 신뢰성이 확보된 전자적 수단을 활용한 보험수익자 의사표시의 확인방법 포\n'
 '- 함)\n'
 '- 6. 기타 보험회사가 필요하다고 인정하는 서류 및 보험수익자가 보험금의 수령에 반\n'
 '- 필요하여 제출하는 서류 려동\n'
 '- 물'),
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
 'indexing': {'chunk_id': 'chunk_000634',
              'chunk_char_len': 271,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
