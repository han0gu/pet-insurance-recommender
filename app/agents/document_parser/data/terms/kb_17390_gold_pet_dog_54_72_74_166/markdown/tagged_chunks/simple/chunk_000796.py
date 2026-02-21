from langchain_core.documents import Document

chunk = Document(
    page_content=('당 과세기간에 만기에 환급되는 금액이 납입보험료를 초과하지 아니하는 보\n'
 '험의 보험계약에 따라 지급하는 다음 각 호의 보험료를 지급한 경우 그 금액\n'
 '의 100분의 12(제1호의 경우에는 100분의 15)에 해당하는 금액을 해당 과세\n'
 '기간의 종합소득산출세액에서 공제한다. 다만, 다음 각 호의 보험료별로 그\n'
 '합계액이 각각 연 100만원을 초과하는 경우 그 초과하는 금액은 각각 없는 것\n'
 '으로 한다.\n'
 '1. 기본공제대상자 중 장애인을 피보험자 또는 수익자로 하는 장애인전용보∙- 험으로서 대통령령으로 정하는 장애인전용보장성보험료'),
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
 'indexing': {'chunk_id': 'chunk_000796',
              'chunk_char_len': 291,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
