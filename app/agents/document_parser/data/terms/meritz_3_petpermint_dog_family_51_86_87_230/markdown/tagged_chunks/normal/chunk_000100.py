from langchain_core.documents import Document

chunk = Document(
    page_content=('일의 다음날부터 그 다음 보험료의 납입최고(독촉)기간까지\n'
 '의 이자(보험계약대출이율을 적용하여 계산)를 더한 금액이\n'
 '해당 보험료가 납입된 것으로 계산한 해약환급금과 계약자\n'
 '에게 지급할 기타 모든 지급금의 합계액에서 계약자의 회사\n'
 '에 대한 모든 채무액을 뺀 금액을 초과하는 경우에는 보험\n'
 '료의 자동대출납입을 더는 할 수 없습니다.\n'
 '\uf000 제1항 및 제2항에 따른 보험료의 자동대출납입 기간은\n'
 '최초 자동대출납입일부터 1년을 한도로 하며 그 이후의 기\n'
 '간에 대한 보험료의 자동대출납입을 위해서는 제1항에 따라\n'
 '재신청을 하여야 합니다.'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'limit', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000100',
              'chunk_char_len': 292,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
