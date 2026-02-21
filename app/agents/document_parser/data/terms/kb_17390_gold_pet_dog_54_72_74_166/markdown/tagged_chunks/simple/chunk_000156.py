from langchain_core.documents import Document

chunk = Document(
    page_content=('- \uf000 제1항의 규정에 의한 대출금과 보험료의 자동대출 납입일의 다음날부터 그 다음 보\n'
 '- 험료의 납입최고(독촉)기간까지의 이자(보험계약대출이율 이내에서 회사가 별도로\n'
 '- 정하는 이율을 적용하여 계산)를 더한 금액이 해당 보험료가 납입된 것으로 계산한\n'
 '- 해약환급금과 계약자에게 지급할 기타 모든 지급금의 합계액에서 계약자의 회사에\n'
 '- 대한 모든 채무액을 뺀 금액을 초과하는 경우에는 보험료의 자동대출납입을 더는\n'
 '- 할 수 없습니다.\n'
 '- \uf000 제1항 및 제2항에 따른 보험료의 자동대출납입 기간은 최초 자동대출납입일부터 1'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000156',
              'chunk_char_len': 291,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
