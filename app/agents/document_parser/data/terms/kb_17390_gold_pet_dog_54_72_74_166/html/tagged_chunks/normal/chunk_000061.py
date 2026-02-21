from langchain_core.documents import Document

chunk = Document(
    page_content=('합니다)을 연단위 복리로 적<br>립한 금액(적립한 금액에서 중도인출액이 있었던 경우에는 그 원금과 이자 합계액 '
 "보</p><br><table id='77' "
 "style='font-size:16px'><thead></thead><tbody><tr><td>을 차감하여 계산한 "
 '금액)을</td><td>만기환급금으로 보험수익자에게</td><td>지급합니다'),
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
 'indexing': {'chunk_id': 'chunk_000061',
              'chunk_char_len': 193,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
