from langchain_core.documents import Document

chunk = Document(
    page_content=('있을 경우 각 계약에 대하여 다른 계약이 없는 것으로 하여 각<br>각 산출한 보상책임액의 합계액이 손해액을 초과할 때에는 아래에 따라 '
 "손해를</p><br><p id='88' data-category='paragraph' "
 "style='font-size:14px'>보상합니다.<br>다른 계약이 없을 때<br>피보험자가 이 계약의 "
 '지급보험금<br>×<br>부담한 위탁비용 다른 계약이 없는 것으로 하여<br>각각 계산한 지급보험금의 합계액</p><br><figure '
 'id=\'89\'><img alt=""'),
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
 'indexing': {'chunk_id': 'chunk_001265',
              'chunk_char_len': 279,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
