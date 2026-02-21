from langchain_core.documents import Document

chunk = Document(
    page_content=('해약환급금 범위 내에서 납입할 보험료를 자동적으로 대출하여 이를 보험료 납입에 충당하는\n'
 '서비스를 말합니다.② 제1항의 규정에 의한 대출금과 보험료의 자동대출납입일의 다음날부터 그 다음 보험\n'
 '료의 납입최고(독촉)기간까지의 이자(보험계약대출이율 이내에서 회사가 별도로 정하\n'
 '는 이율을 적용하여 계산)를 더한 금액이 해당 보험료가 납입된 것으로 계산한 해약환\n'
 '급금과 계약자에게 지급할 기타 모든 지급금의 합계액에서 계약자의 회사에 대한 모\n'
 '든 채무액을 뺀 금액을 초과하는 경우에는 보험료의 자동대출납입을 더는 할 수 없습'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000216',
              'chunk_char_len': 288,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
