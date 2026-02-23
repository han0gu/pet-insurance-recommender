from langchain_core.documents import Document

chunk = Document(
    page_content=('「계약자」라 합니다)가 청약하고 회사가 승낙함으로써 다음 각 호의 조건을 모두 만족\n'
 '하는 보험계약(이하「전환대상계약」이라 합니다)에 대하여 장애인전용보험으로 전환을\n'
 '청약하는 경우에 적용합니다.1. 「소득세법 제59조의4(특별세액공제) 제1항 제2호」에 따라 보험료가 특별세액공제\n'
 '의 대상이 되는 보험【소득세법 제59조의4(특별세액공제)】① 근로소득이 있는 거주자(일용근로자는 제외한다. 이하 이 조에서 같다)가 해당 '
 '과세\n'
 '기간에 만기에 환급되는 금액이 납입보험료를 초과하지 아니하는 보험의 보험계약에'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000196',
              'chunk_char_len': 279,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
