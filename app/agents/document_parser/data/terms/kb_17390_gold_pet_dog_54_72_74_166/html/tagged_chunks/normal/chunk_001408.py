from langchain_core.documents import Document

chunk = Document(
    page_content=('. 이하 이 조에서 같다)가 해<br>당 과세기간에 만기에 환급되는 금액이 납입보험료를 초과하지 아니하는 보<br>험의 보험계약에 따라 '
 '지급하는 다음 각 호의 보험료를 지급한 경우 그 금액<br>의 100분의 12(제1호의 경우에는 100분의 15)에 해당하는 금액을 해당 '
 '과세<br>기간의 종합소득산출세액에서 공제한다. 다만, 다음 각 호의 보험료별로 그<br>합계액이 각각 연 100만원을 초과하는 경우 그 '
 '초과하는 금액은 각각 없는 것<br>으로 한다.<br>1'),
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
 'indexing': {'chunk_id': 'chunk_001408',
              'chunk_char_len': 261,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
