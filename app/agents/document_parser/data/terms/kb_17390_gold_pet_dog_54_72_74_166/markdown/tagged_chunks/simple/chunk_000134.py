from langchain_core.documents import Document

chunk = Document(
    page_content=('- 성별에 해당하는 보험금 및 보험료로 변경합니다. 여기서 "신분증"이란 주민등록\n'
 '- 증이나 운전면허증 등 사진이 붙은 정부기관 발행 신분증을 말합니다.\n'
 '- \uf000 제3항에 따라 보험금 및 보험료를 변경할 때 변경 전후의 계약자적립액 또는 해약\n'
 '- 환급금 등의 차이로 계약자가 추가로 납입하거나 반환받을 금액이 발생할 수 있습\n'
 '- 니다.\n'
 '예 시| ∙ 보험나이 | 계산 |\n'
 '| --- | --- |'),
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
 'indexing': {'chunk_id': 'chunk_000134',
              'chunk_char_len': 219,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
