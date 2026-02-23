from langchain_core.documents import Document

chunk = Document(
    page_content=('. 제14조(계약 전 알릴 의무)에 따라 계약자 또는 피보험자가 회사에 알린 내용<br>이나 건강진단 내용이 보험금 지급사유의 발생에 '
 '영향을 미쳤음을 회사가 증명<br>하는 경우<br>2. 제16조(알릴 의무 위반의 효과)를 준용하여 회사가 보장을 하지 않을 수 '
 '있는<br>경우<br>3. 진단계약에서 보험금 지급사유가 발생할 때까지 진단을 받지 않은 경우'),
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
 'indexing': {'chunk_id': 'chunk_000226',
              'chunk_char_len': 199,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
