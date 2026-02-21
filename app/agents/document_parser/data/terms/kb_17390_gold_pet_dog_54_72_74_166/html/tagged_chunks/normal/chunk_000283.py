from langchain_core.documents import Document

chunk = Document(
    page_content=('. 그러나, 순수보장성보험 등 보험<br>상품의 종류에 따라 보험계약대출이 제한될 수도 있습니다.<br>\uf000 계약자는 제1항에 '
 '따른 보험계약대출금과 그 이자를 언제든지 상환할 수 있으며 상<br>환하지 않은 때에는 회사는 보험금, 해약환급금 등의 지급사유가 발생한 '
 '날에 지급<br>금에서 보험계약대출의 원금과 이자를 차감할 수 있습니다.<br>\uf000 제2항의 규정에도 불구하고 회사는 '
 '제28조(보험료의 납입이 연체되는 경우 납입최<br>고(독촉)와 계약의 해지)에 따라 계약이 해지되는 때에는 즉시 해약환급금에서 '
 '보<br>험계약대출의 원금과'),
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
 'indexing': {'chunk_id': 'chunk_000283',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
