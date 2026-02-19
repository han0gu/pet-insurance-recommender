from langchain_core.documents import Document

chunk = Document(
    page_content=('② 제1항의 규정에 의한 대출금과 보험료의 자동대출납입일의 다음날부터 그 다음 보험 료의 납입최고(독촉)기간까지의 이자(보험계약대출이율 '
 '이내에서 회사가 별도로 정하 는 이율을 적용하여 계산)를 더한 금액이 해당 보험료가 납입된 것으로 계산한 해약환 급금과 계약자에게 지급할 '
 '기타 모든 지급금의 합계액에서 계약자의 회사에 대한 모 든 채무액을 뺀 금액을 초과하는 경우에는 보험료의 자동대출납입을 더는 할 수 없습 '
 '니다.\n'
 '<용어풀이>\n'
 '[보험계약대출이율]'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 67,
         'page': 38},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000105',
              'chunk_char_len': 251,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
