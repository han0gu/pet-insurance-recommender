from langchain_core.documents import Document

chunk = Document(
    page_content=('것으로 계산한<br>해약환급금과 계약자에게 지급할 기타 모든 지급금의 합계액에서 계약자의 회사에<br>대한 모든 채무액을 뺀 금액을 '
 '초과하는 경우에는 보험료의 자동대출납입을 더는<br>할 수 없습니다.<br>\uf000 제1항 및 제2항에 따른 보험료의 자동대출납입 '
 '기간은 최초 자동대출납입일부터 1<br>년을 한도로 하며 그 이후의 기간에 대한 보험료의 자동대출납입을 위해서는 제1항<br>에 따라 '
 '재신청을 하여야 합니다.<br>\uf000 보험료의 자동대출 납입이 행하여진 경우에도 자동대출 납입전 납입최고(독촉)기<br>간이 끝나는 '
 '날의 다음날부터'),
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
 'indexing': {'chunk_id': 'chunk_000234',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
