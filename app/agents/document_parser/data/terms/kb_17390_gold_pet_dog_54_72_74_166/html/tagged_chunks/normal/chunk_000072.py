from langchain_core.documents import Document

chunk = Document(
    page_content=("data-category='list' style='font-size:14px'>\uf000 계약자(보험금 지급사유 발생 후에는 "
 '보험수익자)는 회사의 사업방법서에서 정한<br>바에 따라 보험금의 전부 또는 일부에 대하여 나누어 지급받거나 일시에 '
 '지급받는<br>방법으로 변경할 수 있습니다.<br>\uf000 회사는 제1항에 따라 일시에 지급할 금액을 나누어 지급하는 경우에는 나중에 '
 '지급<br>할 금액에 대하여 평균공시이율을 연단위 복리로 계산한 금액을 더하며, 나누어 지<br>급할 금액을 일시에 지급하는 경우에는 '
 '평균공시이율을 연단위 복리로 할인한'),
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
 'indexing': {'chunk_id': 'chunk_000072',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
