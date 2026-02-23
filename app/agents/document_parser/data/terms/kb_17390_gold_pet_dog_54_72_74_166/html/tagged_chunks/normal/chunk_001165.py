from langchain_core.documents import Document

chunk = Document(
    page_content=('결정된 후 7일(이하 "지급기일"이라 합니다)이 지<br>나도록 보험금을 지급하지 않았을 때에는 지급기일의 다음날부터 '
 '지급일까지의<br>기간에 대하여 "보험금을 지급할 때의 적립이율 계산(【별표2】참조)"에서 정한<br>이율로 계산한 금액을 보험금에 '
 '더하여 지급합니다'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_001165',
              'chunk_char_len': 149,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
