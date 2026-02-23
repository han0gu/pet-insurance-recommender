from langchain_core.documents import Document

chunk = Document(
    page_content=('② 제1항의 규정에 따라 해지하지 않은 계약은 파산선고 후 3개월이 지난 때에는 그 효력\n'
 '을 잃습니다.\n'
 '③ 제1항의 규정에 따라 계약이 해지되거나 제2항의 규정에 따라 계약이 효력을 잃는 경- 18 -우에 회사는 제33조(보험료의 환급)에 '
 '따른 보험료를 계약자에게 지급합니다.제33조(보험료의 환급)① 이 계약이 무효, 효력상실, 해지 또는 소멸된 때에는 다음과 같이 보험료를 '
 '돌려드립니\n'
 '다.1. 계약자, 피보험자 또는 보험수익자의 책임없는 사유에 의하는 경우 : 무효의 경우에'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000101',
              'chunk_char_len': 267,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
