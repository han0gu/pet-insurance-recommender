from langchain_core.documents import Document

chunk = Document(
    page_content=('- 에 있어 지급일까지의 기간에 대한 이자의 계산은 "보험금을 지급할 때의 적립이율 사항\n'
 '- 계산"(【별표2】참조)에 따릅니다.\n'
 '- \uf000 회사는 보험기간이 끝난 때에는 적립부분 순보험료에 대하여 보험료납입일부터 이\n'
 '- 보험의 "보장성-1701 공시이율"(이하 "공시이율"이라 합니다)을 연단위 복리로 적\n'
 '- 립한 금액(적립한 금액에서 중도인출액이 있었던 경우에는 그 원금과 이자 합계액 보\n'
 '| 을 차감하여 계산한 금액)을 | 만기환급금으로 보험수익자에게 | 지급합니다. 통약 |\n'
 '| --- | --- | --- |'),
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
 'indexing': {'chunk_id': 'chunk_000036',
              'chunk_char_len': 286,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
