from langchain_core.documents import Document

chunk = Document(
    page_content=('장애인전용보험전환 특별약관\n'
 '제1조(특별약관의 적용범위)\n'
 '① 이 특별약관은 보험회사(이하「회사」라 합니다)가 정한 방법에 따라 보험계약자(이하 「계약자」라 합니다)가 청약하고 회사가 승낙함으로써 '
 '다음 각 호의 조건을 모두 만족 하는 보험계약(이하「전환대상계약」이라 합니다)에 대하여 장애인전용보험으로 전환을 청약하는 경우에 '
 '적용합니다.\n'
 '1. 「소득세법 제59조의4(특별세액공제) 제1항 제2호」에 따라 보험료가 특별세액공제 의 대상이 되는 보험\n'
 '【소득세법 제59조의4(특별세액공제)】'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 44},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000235',
              'chunk_char_len': 269,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
