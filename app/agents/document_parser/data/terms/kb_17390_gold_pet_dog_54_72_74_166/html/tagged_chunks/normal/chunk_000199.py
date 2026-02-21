from langchain_core.documents import Document

chunk = Document(
    page_content=('. 그러나 계약자<br>가 사망한 경우 그 승계인이 보험수익자 변경에 관한 권리를 행사할 수 있다는 별<br>도의 약정이 있는 경우에는 '
 '승계받은 계약자가 보험수익자를 변경할 수 있습니다.<br>\uf000 회사는 제1항에 따라 계약자를 변경한 경우, 변경된 계약자에게 '
 '보험증권 및 약관<br>을 교부하고 변경된 계약자가 요청하는 경우 약관의 중요한 내용을 설명하여 드립<br>니다.<br>\uf000 '
 '제1항에 따라 위험이 증가하거나 감소되는 경우 납입보험료가 변경될 수 있으며,<br>계약내용 변경시점 이후 잔여 보험기간의 보장을 위한 '
 '재원인 계약자적립액'),
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
 'indexing': {'chunk_id': 'chunk_000199',
              'chunk_char_len': 300,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
