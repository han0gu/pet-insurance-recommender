from langchain_core.documents import Document

chunk = Document(
    page_content=('諾)함으로써 다음 각 호의 조건을 모두 만족하는 보험계약(이하 "전환대상계약"이\n'
 '라 합니다)에 대하여 장애인전용보험으로 전환을 청약하는 경우에 적용합니다.관 련 법 규 소득세법1. 소득세법 제59조의4(특별세액공제) '
 '제1항 제2호에 따라 보험료가 특별세액공\n'
 '제의 대상이 되는 보험# ∙소득세법 제59조의 4(특별세액공제)근로소득이 있는 거주자(일용근로자는 제외한다. 이하 이 조에서 같다)가 '
 '해\n'
 '당 과세기간에 만기에 환급되는 금액이 납입보험료를 초과하지 아니하는 보'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000795',
              'chunk_char_len': 260,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
