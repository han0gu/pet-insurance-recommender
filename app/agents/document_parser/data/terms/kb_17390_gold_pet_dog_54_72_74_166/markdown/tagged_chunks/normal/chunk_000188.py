from langchain_core.documents import Document

chunk = Document(
    page_content=('- 의 잔액을 기준으로 합니다)의 80% 범위 내에서 회사가 정한 방법에 따라 중도인출\n'
 '- 을 할 수 있습니다. 단, 중도인출은 보험기간 내에 한하며, 매 보험년도마다 12회\n'
 '- 에 한합니다.\n'
 '- \uf000 제1항의 중도인출의 총 누적액은 중도인출을 한번도 지급하지 않았을 경우의 기본\n'
 '- 계약 해약환급금과 적립부분 해약환급금 중 적은 금액의 80%를 한도로 합니다.\n'
 '- 용 어 풀 이 보험년도\n'
 '| 보험계약일로부터 | 다음 해의 해당일 전일까지 매1년 |  | 보험계약 단위의 연도를 |\n'
 '| --- | --- | --- | --- |'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'limit', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000188',
              'chunk_char_len': 294,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
