from langchain_core.documents import Document

chunk = Document(
    page_content=('에는 그 장애기간 동안은 이를 다시 제출하지 않을 수 있습니다. 물\n'
 '\uf000 제1항에 따라 제출한 장애인증명서의 장애기간이 변경되는 경우 계약자는 이를 회\n'
 '사에 알리고 변경된 장애기간이 기재된 장애인증명서를 제출하여야 합니다.\n'
 '제- \n'
 '# 제3조(장애인전용보험으로의 전환)# \uf000 회사는 이 특별약관이 부가된- 제1항 제1호에 해당하는 장애인전용보험으로 전환하여 '
 '드립니다.\n'
 '- \uf000 제1항에 따라 전환대상계약이 장애인전용보험으로 전환된 후부터 납입된 전환대상 약'),
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
 'indexing': {'chunk_id': 'chunk_000809',
              'chunk_char_len': 252,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
