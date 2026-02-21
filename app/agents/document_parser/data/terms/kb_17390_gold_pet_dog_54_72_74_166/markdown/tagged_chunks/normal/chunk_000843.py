from langchain_core.documents import Document

chunk = Document(
    page_content=('- 8) ‘뚜렷한 시야 장해’라 함은 한 눈의 시야 범위가 정상시야 범위의 60%\n'
 '- 이하로 제한된 경우를 말한다. 이 경우 시야검사는 공인된 시야검사방법\n'
 '- 으로 측정하며, 시야장해 평가 시 자동시야검사계(골드만 시야검사)를\n'
 '- 이용하여 8방향 시야범위 합계를 정상범위와 비교하여 평가한다.\n'
 '- 9) ‘눈꺼풀에 뚜렷한 결손을 남긴 때’라 함은 눈꺼풀의 결손으로 눈을 감\n'
 '- 았을 때 각막(검은 자위)이 완전히 덮이지 않는 경우를 말한다.\n'
 '- 10) ‘눈꺼풀에 뚜렷한 운동장해를 남긴 때’ 라 함은 눈을 떴을 때 동공을'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'eye']},
 'indexing': {'chunk_id': 'chunk_000843',
              'chunk_char_len': 290,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
